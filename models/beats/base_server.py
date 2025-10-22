#!/usr/bin/env python3
"""
BEATs OpenAI-Compatible Server (Simplified)
A simplified server that wraps BEATs audio classification model in OpenAI API format
Only handles audio URLs (local paths), no text processing
"""

import asyncio
import json
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
import uvicorn

# BEATs imports
from BEATs import BEATs, BEATsConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioMessage(BaseModel):
    """Audio message content"""

    type: str = Field('audio_url', description='Type of content')
    audio_url: Dict[str, str] = Field(..., description='Audio URL information')


class Message(BaseModel):
    """Chat message"""

    role: str = Field(..., description='Role of the message sender')
    content: List[AudioMessage] = Field(..., description='List of audio messages')


class ResponseMessage(BaseModel):
    """Response message"""

    role: str = Field(..., description='Role of the message sender')
    content: str = Field(..., description='Message content')


class ChatCompletionRequest(BaseModel):
    """Chat completion request"""

    model: str = Field(..., description='Model name')
    messages: List[Message] = Field(..., description='List of messages')


class Choice(BaseModel):
    """Response choice"""

    index: int
    message: ResponseMessage
    finish_reason: str = 'stop'


class Usage(BaseModel):
    """Token usage information"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Chat completion response"""

    id: str
    object: str = 'chat.completion'
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelInfo(BaseModel):
    """Model information"""

    id: str
    object: str = 'model'
    created: int
    owned_by: str = 'beats-classifier'


class ModelsResponse(BaseModel):
    """Models list response"""

    object: str = 'list'
    data: List[ModelInfo]


class BEATsClassifier:
    """BEATs audio classifier wrapper"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.label_dict = None
        self.load_model()

    def load_model(self):
        """Load BEATs model from checkpoint"""
        try:
            logger.info(f"Loading BEATs model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            config = BEATsConfig(checkpoint['cfg'])
            self.model = BEATs(config)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.model.eval()

            self.label_dict = checkpoint.get('label_dict', {})
            logger.info(f"Model loaded successfully with {len(self.label_dict)} classes")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """Preprocess audio file for BEATs model"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)

            # Normalize
            waveform = waveform / torch.max(torch.abs(waveform))

            return waveform.squeeze(0)  # Remove channel dimension

        except Exception as e:
            logger.error(f"Failed to preprocess audio {audio_path}: {str(e)}")
            raise

    def classify_audio_batch(self, audio_paths: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Classify multiple audio files in batch and return top-k predictions for each"""
        if not audio_paths:
            return []

        try:
            # Preprocess all audio files
            audio_tensors = []
            max_length = 0

            for audio_path in audio_paths:
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                audio_tensor = self.preprocess_audio(audio_path)
                audio_tensors.append(audio_tensor)
                max_length = max(max_length, audio_tensor.shape[0])

            # Pad all audio tensors to the same length
            batch_audio = []
            batch_padding_mask = []

            for audio_tensor in audio_tensors:
                if audio_tensor.shape[0] < max_length:
                    # Pad with zeros
                    padded_audio = torch.cat([audio_tensor, torch.zeros(max_length - audio_tensor.shape[0])])
                else:
                    padded_audio = audio_tensor

                batch_audio.append(padded_audio)

                # Create padding mask (False for real audio, True for padding)
                padding_mask = torch.zeros(max_length).bool()
                padding_mask[audio_tensor.shape[0] :] = True
                batch_padding_mask.append(padding_mask)

            # Stack into batch tensors
            batch_audio = torch.stack(batch_audio)
            batch_padding_mask = torch.stack(batch_padding_mask)

            # Move to device
            batch_audio = batch_audio.to(self.device)
            batch_padding_mask = batch_padding_mask.to(self.device)

            # Extract features and get predictions for the entire batch
            with torch.no_grad():
                probs = self.model.extract_features(batch_audio, padding_mask=batch_padding_mask)[0]

                # Get top-k predictions for each sample in the batch
                batch_results = []
                top_k_probs, top_k_indices = probs.topk(k=top_k)

                for i in range(len(audio_paths)):
                    results = []
                    for prob, idx in zip(top_k_probs[i], top_k_indices[i]):
                        label = self.label_dict.get(idx.item(), f"class_{idx.item()}")
                        results.append({'label': label, 'probability': prob.item(), 'class_id': idx.item()})
                    batch_results.append(results)

                return batch_results

        except Exception as e:
            logger.error(f"Failed to classify audio batch: {str(e)}")
            raise


# ----------------------------
# Micro-batching implementation
# ----------------------------


@dataclass
class _BatchReq:
    req_id: int
    audio_paths: List[str]
    future: asyncio.Future


class MicroBatcher:
    def __init__(self, classifier: BEATsClassifier, max_batch_size: int = 32, max_wait_ms: int = 30):
        self.classifier = classifier
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: deque[_BatchReq] = deque()
        self._req_id = 0
        self._lock = asyncio.Lock()
        self._runner_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self):
        self._runner_task = asyncio.create_task(self._runner())

    async def stop(self):
        self._stop.set()
        if self._runner_task:
            await self._runner_task

    async def submit(self, audio_paths: List[str]) -> List[List[Dict[str, Any]]]:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._req_id += 1
            self.queue.append(_BatchReq(self._req_id, audio_paths, fut))
        return await fut

    async def _runner(self):
        while not self._stop.is_set():
            await asyncio.sleep(self.max_wait_ms / 1000.0)

            batch: List[_BatchReq] = []
            async with self._lock:
                if not self.queue:
                    continue
                while self.queue and len(batch) < self.max_batch_size:
                    batch.append(self.queue.popleft())

            if not batch:
                continue

            concat_audio: List[str] = []
            sizes: List[int] = []
            for br in batch:
                concat_audio.extend(br.audio_paths)
                sizes.append(len(br.audio_paths))

            try:
                all_pred = self.classifier.classify_audio_batch(concat_audio)
                offset = 0
                for br, n in zip(batch, sizes):
                    result_slice = all_pred[offset : offset + n]
                    offset += n
                    if not br.future.cancelled():
                        br.future.set_result(result_slice)
            except Exception as e:
                for br in batch:
                    if not br.future.cancelled():
                        br.future.set_exception(e)


# Global instances
classifier = None
batcher: Optional[MicroBatcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model on startup"""
    global classifier, batcher

    model_path = os.getenv('BEATS_MODEL_PATH', '/path/to/model.pt')
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        logger.error('Please set BEATS_MODEL_PATH environment variable')
        yield
        return

    try:
        classifier = BEATsClassifier(model_path, device)
        # Configure micro-batching from env (optional)
        max_batch_size = int(os.getenv('MB_MAX_BATCH_SIZE', '8'))
        max_wait_ms = int(os.getenv('MB_MAX_WAIT_MS', '20'))
        batcher = MicroBatcher(classifier, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
        await batcher.start()
        logger.info(
            f"BEATs classifier and MicroBatcher initialized (max_batch_size={max_batch_size}, max_wait_ms={max_wait_ms})"
        )
    except Exception as e:
        logger.error(f"Failed to initialize classifier/batcher: {str(e)}")

    yield

    # Cleanup code here if needed
    if batcher:
        await batcher.stop()
    logger.info('Shutting down BEATs server')


# Initialize FastAPI app
app = FastAPI(
    title='BEATs Audio Classification Server',
    description='OpenAI-compatible API for BEATs audio classification',
    version='1.0.0',
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/v1/models')
async def list_models():
    """List available models"""
    return ModelsResponse(data=[ModelInfo(id='beats-classifier', created=1672531200, owned_by='beats-classifier')])


@app.post('/v1/chat/completions')
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests"""
    if classifier is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    if batcher is None:
        raise HTTPException(status_code=503, detail='Batcher not ready')

    try:
        # Extract audio URLs from messages
        audio_urls = []
        for message in request.messages:
            for content in message.content:
                if content.type == 'audio_url':
                    audio_urls.append(content.audio_url['url'])

        if not audio_urls:
            raise HTTPException(status_code=400, detail='No audio URLs found in request')

        # Submit to micro-batcher to aggregate with other in-flight requests
        batch_predictions = await batcher.submit(audio_urls)

        # Format response as structured JSON
        classification_results = []
        for i, (audio_url, predictions) in enumerate(zip(audio_urls, batch_predictions)):
            audio_result = {
                'audio_index': i + 1,
                'audio_file': os.path.basename(audio_url),
                'audio_path': audio_url,
                'predictions': predictions,
                'top_prediction': predictions[0] if predictions else None,
            }
            classification_results.append(audio_result)

        # Create structured response
        structured_response = {
            'classification_results': classification_results,
            'summary': {'total_audio_files': len(audio_urls), 'successfully_processed': len(classification_results)},
        }

        # Convert to JSON string for response
        response_text = json.dumps(structured_response, indent=2, ensure_ascii=False)

        # Create response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{hash(str(request))}",
            created=int(asyncio.get_event_loop().time()),
            model=request.model,
            choices=[
                Choice(index=0, message=ResponseMessage(role='assistant', content=response_text), finish_reason='stop')
            ],
            usage=Usage(
                prompt_tokens=len(str(request)),
                completion_tokens=len(response_text),
                total_tokens=len(str(request)) + len(response_text),
            ),
        )

        return response

    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'model_loaded': classifier is not None}


@app.get('/')
async def root():
    """Root endpoint"""
    return {
        'message': 'BEATs Audio Classification Server (Simplified)',
        'version': '1.0.0',
        'endpoints': {'models': '/v1/models', 'chat_completions': '/v1/chat/completions', 'health': '/health'},
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BEATs OpenAI Server (Simplified)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8901, help='Port to bind to')
    parser.add_argument(
        '--model-path',
        default='/inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/cargoflow/src/cargoflow/llms/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
        help='Path to BEATs model checkpoint',
    )
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')

    args = parser.parse_args()

    # Set environment variables
    if args.model_path:
        os.environ['BEATS_MODEL_PATH'] = args.model_path

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    os.environ['DEVICE'] = device

    logger.info(f"Starting BEATs server on {args.host}:{args.port}")
    logger.info(f"Model path: {os.getenv('BEATS_MODEL_PATH')}")
    logger.info(f"Device: {device}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level='info')
