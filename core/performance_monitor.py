"""
Performance monitoring and metrics collection for the multi-agent system.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import json

@dataclass
class PerformanceMetric:
    """Represents a performance metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    agent_id: Optional[str] = None
    task_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_metrics_per_type: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_aggregation = datetime.now()
        self.aggregation_interval = timedelta(minutes=5)
    
    def record_metric(
        self, 
        metric_name: str, 
        value: float,
        agent_id: Optional[str] = None,
        task_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            agent_id=agent_id,
            task_type=task_type,
            metadata=metadata or {}
        )
        
        self.metrics[metric_name].append(metric)
        
        # Trigger aggregation if needed
        if datetime.now() - self.last_aggregation >= self.aggregation_interval:
            self._aggregate_metrics()
    
    def get_metrics(
        self, 
        metric_name: str, 
        since: Optional[datetime] = None,
        agent_id: Optional[str] = None
    ) -> List[PerformanceMetric]:
        """Get metrics with optional filtering."""
        metrics = list(self.metrics.get(metric_name, []))
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        if agent_id:
            metrics = [m for m in metrics if m.agent_id == agent_id]
        
        return metrics
    
    def get_aggregated_metrics(self, metric_name: str) -> Dict[str, Any]:
        """Get aggregated metrics for a specific metric type."""
        return self.aggregated_metrics.get(metric_name, {})
    
    def _aggregate_metrics(self) -> None:
        """Aggregate metrics for better performance analysis."""
        for metric_name, metric_list in self.metrics.items():
            if not metric_list:
                continue
            
            values = [m.value for m in metric_list]
            self.aggregated_metrics[metric_name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else 0,
                "last_updated": datetime.now()
            }
        
        self.last_aggregation = datetime.now()

class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        self.system_performance: Dict[str, Any] = {}
        self.running = False
        self.start_time = datetime.now()
    
    async def start(self) -> None:
        """Start the performance monitoring system."""
        self.running = True
        self.start_time = datetime.now()
        
        # Start background monitoring tasks
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._generate_performance_reports())
    
    def stop(self) -> None:
        """Stop the performance monitoring system."""
        self.running = False
    
    def record_task_start(self, task_id: str, agent_id: str, task_type: str) -> None:
        """Record when a task starts execution."""
        self.metrics_collector.record_metric(
            "task_started",
            1,
            agent_id=agent_id,
            task_type=task_type,
            metadata={"task_id": task_id}
        )
    
    def record_task_completion(
        self, 
        task_id: str, 
        agent_id: str, 
        task_type: str, 
        duration: float,
        success: bool
    ) -> None:
        """Record task completion metrics."""
        # Record task duration
        self.metrics_collector.record_metric(
            "task_duration",
            duration,
            agent_id=agent_id,
            task_type=task_type,
            metadata={"task_id": task_id, "success": success}
        )
        
        # Record task completion
        self.metrics_collector.record_metric(
            "task_completed",
            1,
            agent_id=agent_id,
            task_type=task_type,
            metadata={"task_id": task_id, "success": success}
        )
        
        # Record success/failure
        metric_name = "task_success" if success else "task_failure"
        self.metrics_collector.record_metric(
            metric_name,
            1,
            agent_id=agent_id,
            task_type=task_type,
            metadata={"task_id": task_id}
        )
    
    def record_agent_metric(
        self, 
        agent_id: str, 
        metric_name: str, 
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an agent-specific metric."""
        self.metrics_collector.record_metric(
            metric_name,
            value,
            agent_id=agent_id,
            metadata=metadata
        )
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent."""
        # Get recent metrics for this agent
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        task_durations = self.metrics_collector.get_metrics(
            "task_duration", since=one_hour_ago, agent_id=agent_id
        )
        
        task_successes = self.metrics_collector.get_metrics(
            "task_success", since=one_hour_ago, agent_id=agent_id
        )
        
        task_failures = self.metrics_collector.get_metrics(
            "task_failure", since=one_hour_ago, agent_id=agent_id
        )
        
        total_tasks = len(task_successes) + len(task_failures)
        success_rate = len(task_successes) / total_tasks if total_tasks > 0 else 0
        
        avg_duration = (
            sum(m.value for m in task_durations) / len(task_durations)
            if task_durations else 0
        )
        
        return {
            "agent_id": agent_id,
            "total_tasks_1h": total_tasks,
            "success_rate": success_rate,
            "avg_task_duration": avg_duration,
            "tasks_per_hour": total_tasks,
            "last_task_time": task_durations[-1].timestamp if task_durations else None
        }
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        # Get all task metrics from last hour
        all_tasks = (
            self.metrics_collector.get_metrics("task_success", since=one_hour_ago) +
            self.metrics_collector.get_metrics("task_failure", since=one_hour_ago)
        )
        
        task_durations = self.metrics_collector.get_metrics("task_duration", since=one_hour_ago)
        
        # Calculate system-wide metrics
        total_tasks = len(all_tasks)
        successful_tasks = len(self.metrics_collector.get_metrics("task_success", since=one_hour_ago))
        
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        avg_duration = (
            sum(m.value for m in task_durations) / len(task_durations)
            if task_durations else 0
        )
        
        # Calculate throughput (tasks per minute)
        throughput = total_tasks / 60 if total_tasks > 0 else 0
        
        # System uptime
        uptime = datetime.now() - self.start_time
        
        return {
            "total_tasks_1h": total_tasks,
            "success_rate": success_rate,
            "avg_task_duration": avg_duration,
            "throughput_per_minute": throughput,
            "system_uptime": uptime.total_seconds(),
            "timestamp": datetime.now()
        }
    
    def get_task_type_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics grouped by task type."""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        # Group metrics by task type
        task_types = defaultdict(lambda: {"successes": [], "failures": [], "durations": []})
        
        for metric in self.metrics_collector.get_metrics("task_success", since=one_hour_ago):
            if metric.task_type:
                task_types[metric.task_type]["successes"].append(metric)
        
        for metric in self.metrics_collector.get_metrics("task_failure", since=one_hour_ago):
            if metric.task_type:
                task_types[metric.task_type]["failures"].append(metric)
        
        for metric in self.metrics_collector.get_metrics("task_duration", since=one_hour_ago):
            if metric.task_type:
                task_types[metric.task_type]["durations"].append(metric)
        
        # Calculate metrics for each task type
        result = {}
        for task_type, metrics in task_types.items():
            total_tasks = len(metrics["successes"]) + len(metrics["failures"])
            success_rate = len(metrics["successes"]) / total_tasks if total_tasks > 0 else 0
            avg_duration = (
                sum(m.value for m in metrics["durations"]) / len(metrics["durations"])
                if metrics["durations"] else 0
            )
            
            result[task_type] = {
                "total_tasks": total_tasks,
                "success_rate": success_rate,
                "avg_duration": avg_duration,
                "throughput_per_hour": total_tasks
            }
        
        return result
    
    async def _collect_system_metrics(self) -> None:
        """Background task to collect system-level metrics."""
        while self.running:
            try:
                # Record system uptime
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.metrics_collector.record_metric("system_uptime", uptime)
                
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _generate_performance_reports(self) -> None:
        """Background task to generate periodic performance reports."""
        while self.running:
            try:
                # Update cached performance data
                self.system_performance = self.get_system_performance()
                
                await asyncio.sleep(300)  # Update every 5 minutes
            except Exception as e:
                print(f"Error generating performance reports: {e}")
                await asyncio.sleep(300)
