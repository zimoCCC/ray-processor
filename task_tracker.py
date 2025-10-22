# -*- coding: utf-8 -*-
"""
任务跟踪数据库管理模块
用于实现断点续跑功能
"""
import sqlite3
import time
import logging
from typing import List, Set, Dict, Any
import os

logger = logging.getLogger(__name__)


class TaskTracker:
    """任务跟踪器，管理SQLite数据库中的任务状态"""
    
    def __init__(self, db_path: str):
        """
        初始化任务跟踪器
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        # 启用WAL模式以提高并发性能
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 创建任务状态表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_status (
                    task_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'unallocated'
                        CHECK (status IN ('unallocated', 'allocated', 'completed')),
                    allocated_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    worker_id TEXT
                )
            ''')
            
            # 创建索引以提高查询性能
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON task_status(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_allocated_at ON task_status(allocated_at)')
            
            # 创建元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()
    
    def init_tasks(self, total_tasks: int):
        """
        初始化所有任务为unallocated状态
        
        Args:
            total_tasks: 总任务数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            conn.execute("BEGIN IMMEDIATE")
            
            # 初始化所有任务为unallocated状态
            for task_id in range(total_tasks):
                cursor.execute('''
                    INSERT OR IGNORE INTO task_status (task_id, status) 
                    VALUES (?, 'unallocated')
                ''', (task_id,))
            
            # 更新元数据
            cursor.execute('''
                INSERT OR REPLACE INTO task_metadata (key, value) 
                VALUES ('total_tasks', ?)
            ''', (str(total_tasks),))
            
            cursor.execute('''
                INSERT OR REPLACE INTO task_metadata (key, value) 
                VALUES ('initialized_at', ?)
            ''', (str(int(time.time())),))
            
            conn.commit()
            logger.info(f"Initialized {total_tasks} tasks")
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error initializing tasks: {e}")
            raise
        finally:
            conn.close()
    
    def get_unallocated_tasks(self, batch_size: int) -> List[int]:
        """
        获取未分配的任务ID列表
        
        Args:
            batch_size: 批次大小
            
        Returns:
            未分配的任务ID列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT task_id FROM task_status 
                WHERE status = 'unallocated' 
                ORDER BY task_id 
                LIMIT ?
            ''', (batch_size,))
            
            task_ids = [row[0] for row in cursor.fetchall()]
            return task_ids
            
        except sqlite3.Error as e:
            logger.error(f"Error getting unallocated tasks: {e}")
            return []
        finally:
            conn.close()
    
    def mark_tasks_allocated(self, task_ids: List[int], worker_id: str):
        """
        标记任务为已分配状态
        
        Args:
            task_ids: 任务ID列表
            worker_id: 工作进程ID
        """
        if not task_ids:
            return
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            conn.execute("BEGIN IMMEDIATE")
            current_time = time.time()
            
            for task_id in task_ids:
                cursor.execute('''
                    UPDATE task_status 
                    SET status = 'allocated', allocated_at = ?, worker_id = ?
                    WHERE task_id = ? AND status = 'unallocated'
                ''', (current_time, worker_id, task_id))
            
            conn.commit()
            logger.debug(f"Marked {len(task_ids)} tasks as allocated to {worker_id}")
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error marking tasks allocated: {e}")
            raise
        finally:
            conn.close()
    
    def mark_tasks_completed(self, task_ids: List[int]):
        """
        标记任务为已完成状态
        
        Args:
            task_ids: 任务ID列表
        """
        if not task_ids:
            return
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            conn.execute("BEGIN IMMEDIATE")
            current_time = time.time()
            
            for task_id in task_ids:
                cursor.execute('''
                    UPDATE task_status 
                    SET status = 'completed', completed_at = ?
                    WHERE task_id = ? AND status = 'allocated'
                ''', (current_time, task_id))
            
            conn.commit()
            logger.debug(f"Marked {len(task_ids)} tasks as completed")
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error marking tasks completed: {e}")
            raise
        finally:
            conn.close()
    
    def get_completed_tasks(self) -> Set[int]:
        """
        获取已完成的任务ID集合
        
        Returns:
            已完成的任务ID集合
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT task_id FROM task_status WHERE status = "completed"')
            completed_ids = {row[0] for row in cursor.fetchall()}
            return completed_ids
            
        except sqlite3.Error as e:
            logger.error(f"Error getting completed tasks: {e}")
            return set()
        finally:
            conn.close()
    
    def get_progress_stats(self) -> Dict[str, int]:
        """
        获取进度统计信息
        
        Returns:
            状态统计字典
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM task_status
                GROUP BY status
            ''')
            
            stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 确保所有状态都存在
            for status in ['unallocated', 'allocated', 'completed']:
                stats.setdefault(status, 0)
            
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Error getting progress stats: {e}")
            return {'unallocated': 0, 'allocated': 0, 'completed': 0}
        finally:
            conn.close()
    
    def reset_incomplete_allocations(self):
        """重置未完成的分配任务为unallocated状态"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            conn.execute("BEGIN IMMEDIATE")
            
            # 检查是否有未完成的分配
            cursor.execute("SELECT COUNT(*) FROM task_status WHERE status = 'allocated'")
            incomplete_count = cursor.fetchone()[0]
            
            if incomplete_count > 0:
                # 重置为unallocated
                cursor.execute('''
                    UPDATE task_status
                    SET status = 'unallocated', allocated_at = NULL, worker_id = NULL
                    WHERE status = 'allocated'
                ''')
                logger.info(f"Reset {incomplete_count} incomplete allocations to unallocated")
            
            conn.commit()
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error resetting incomplete allocations: {e}")
            raise
        finally:
            conn.close()
    
    def is_complete(self) -> bool:
        """检查是否所有任务都已完成"""
        stats = self.get_progress_stats()
        return stats['completed'] > 0 and stats['unallocated'] == 0 and stats['allocated'] == 0
    
    def cleanup(self):
        """清理数据库连接"""
        # SQLite连接会在使用后自动关闭，这里可以添加其他清理逻辑
        pass


# 兼容性函数，保持与原有代码的兼容性
def init_task_tracking(db_path: str, total_tasks: int):
    """初始化任务跟踪数据库（兼容性函数）"""
    tracker = TaskTracker(db_path)
    tracker.init_tasks(total_tasks)


def get_unallocated_tasks(db_path: str, batch_size: int) -> List[int]:
    """获取未分配的任务（兼容性函数）"""
    tracker = TaskTracker(db_path)
    return tracker.get_unallocated_tasks(batch_size)


def mark_tasks_allocated(db_path: str, task_ids: List[int], worker_id: str):
    """标记任务为已分配（兼容性函数）"""
    tracker = TaskTracker(db_path)
    tracker.mark_tasks_allocated(task_ids, worker_id)


def mark_tasks_completed(db_path: str, task_ids: List[int]):
    """标记任务为已完成（兼容性函数）"""
    tracker = TaskTracker(db_path)
    tracker.mark_tasks_completed(task_ids)


def get_completed_tasks(db_path: str) -> Set[int]:
    """获取已完成的任务（兼容性函数）"""
    tracker = TaskTracker(db_path)
    return tracker.get_completed_tasks()


def get_progress_stats(db_path: str) -> Dict[str, int]:
    """获取进度统计（兼容性函数）"""
    tracker = TaskTracker(db_path)
    return tracker.get_progress_stats()
