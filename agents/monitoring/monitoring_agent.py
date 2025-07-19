"""
Monitoring Agents - Track performance metrics and system health.
"""

import asyncio
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from core.base_agent import BaseAgent, Task
from collections import defaultdict, deque
import json

class MonitoringAgent(BaseAgent):
    """Agent specialized in system monitoring and health tracking."""
    
    def __init__(self, agent_id: str):
        capabilities = [
            "system_monitoring",
            "performance_tracking",
            "health_checks",
            "alert_management",
            "log_analysis",
            "resource_monitoring",
            "uptime_tracking",
            "anomaly_detection"
        ]
        super().__init__(agent_id, "monitoring", capabilities)
        
        # Monitoring data storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = []
        self.monitored_systems = {}
        self.alert_rules = self._setup_default_alert_rules()
        self.monitoring_active = False
    
    async def setup_agent(self) -> None:
        """Setup the monitoring agent with tools and configurations."""
        tools = [
            Tool(
                name="monitor_system_health",
                description="Monitor overall system health and performance",
                func=self._monitor_system_health
            ),
            Tool(
                name="track_resource_usage",
                description="Track CPU, memory, and disk usage",
                func=self._track_resource_usage
            ),
            Tool(
                name="analyze_logs",
                description="Analyze system and application logs",
                func=self._analyze_logs
            ),
            Tool(
                name="check_service_status",
                description="Check the status of specific services",
                func=self._check_service_status
            ),
            Tool(
                name="generate_alerts",
                description="Generate alerts based on monitoring data",
                func=self._generate_alerts
            ),
            Tool(
                name="create_dashboard",
                description="Create monitoring dashboards",
                func=self._create_dashboard
            ),
            Tool(
                name="analyze_trends",
                description="Analyze performance trends over time",
                func=self._analyze_trends
            ),
            Tool(
                name="monitor_network",
                description="Monitor network connectivity and performance",
                func=self._monitor_network
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a system monitoring and observability expert. Your role is to:
            
            1. Monitor system health and performance metrics continuously
            2. Track resource usage and identify bottlenecks
            3. Analyze logs and identify issues or patterns
            4. Generate alerts for critical system events
            5. Create comprehensive monitoring dashboards
            6. Detect anomalies and performance degradation
            7. Provide recommendations for system optimization
            
            Always prioritize system stability and proactive issue detection.
            Provide clear, actionable insights and alerts.
            """),
            ("user", "{input}"),
            ("assistant", "I'll monitor that system and set up comprehensive tracking. Let me gather the current metrics."),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process monitoring related tasks."""
        task_type = task.type
        task_data = task.data
        
        if task_type == "system_monitoring":
            return await self._handle_system_monitoring(task_data)
        elif task_type == "performance_tracking":
            return await self._handle_performance_tracking(task_data)
        elif task_type == "health_checks":
            return await self._handle_health_checks(task_data)
        elif task_type == "alert_management":
            return await self._handle_alert_management(task_data)
        elif task_type == "log_analysis":
            return await self._handle_log_analysis(task_data)
        elif task_type == "resource_monitoring":
            return await self._handle_resource_monitoring(task_data)
        elif task_type == "uptime_tracking":
            return await self._handle_uptime_tracking(task_data)
        elif task_type == "anomaly_detection":
            return await self._handle_anomaly_detection(task_data)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def _handle_system_monitoring(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive system monitoring."""
        systems = data.get("systems", ["localhost"])
        monitoring_interval = data.get("interval_seconds", 60)
        
        # Start monitoring if not already active
        if not self.monitoring_active:
            asyncio.create_task(self._continuous_monitoring(monitoring_interval))
            self.monitoring_active = True
        
        current_status = self._monitor_system_health(systems)
        
        return {
            "task_type": "system_monitoring",
            "systems_monitored": systems,
            "monitoring_interval": monitoring_interval,
            "current_status": current_status,
            "monitoring_started": self.monitoring_active,
            "monitored_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_performance_tracking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance metrics tracking."""
        metrics = data.get("metrics", ["cpu", "memory", "disk"])
        duration_hours = data.get("duration_hours", 1)
        
        performance_data = self._track_resource_usage(metrics, duration_hours)
        
        return {
            "task_type": "performance_tracking",
            "metrics_tracked": metrics,
            "duration_hours": duration_hours,
            "performance_data": performance_data,
            "monitored_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_health_checks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check operations."""
        services = data.get("services", [])
        health_check_type = data.get("type", "basic")
        
        health_results = []
        for service in services:
            result = self._check_service_status(service, health_check_type)
            health_results.append(result)
        
        overall_health = "healthy" if all(r["status"] == "healthy" for r in health_results) else "unhealthy"
        
        return {
            "task_type": "health_checks",
            "overall_health": overall_health,
            "services_checked": len(services),
            "health_results": health_results,
            "monitored_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_alert_management(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle alert generation and management."""
        alert_config = data.get("alert_config", {})
        check_interval = data.get("check_interval", 300)  # 5 minutes
        
        alerts_generated = self._generate_alerts(alert_config)
        
        return {
            "task_type": "alert_management",
            "alerts_generated": len(alerts_generated),
            "active_alerts": len([a for a in self.alerts if a["status"] == "active"]),
            "alert_details": alerts_generated,
            "monitored_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_log_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle log analysis tasks."""
        log_sources = data.get("log_sources", [])
        analysis_type = data.get("analysis_type", "error_detection")
        time_range = data.get("time_range_hours", 24)
        
        analysis_result = self._analyze_logs(log_sources, analysis_type, time_range)
        
        return {
            "task_type": "log_analysis",
            "log_sources": log_sources,
            "analysis_type": analysis_type,
            "time_range_hours": time_range,
            "analysis_result": analysis_result,
            "monitored_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_resource_monitoring(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource usage monitoring."""
        resource_types = data.get("resource_types", ["cpu", "memory", "disk", "network"])
        
        resource_data = {}
        for resource_type in resource_types:
            resource_data[resource_type] = self._get_resource_metrics(resource_type)
        
        return {
            "task_type": "resource_monitoring",
            "resource_types": resource_types,
            "resource_data": resource_data,
            "monitored_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_uptime_tracking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle uptime tracking for services."""
        services = data.get("services", [])
        
        uptime_data = {}
        for service in services:
            uptime_data[service] = self._calculate_uptime(service)
        
        return {
            "task_type": "uptime_tracking",
            "services": services,
            "uptime_data": uptime_data,
            "monitored_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_anomaly_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle anomaly detection in metrics."""
        metric_name = data.get("metric_name", "cpu_usage")
        sensitivity = data.get("sensitivity", "medium")
        
        anomalies = self._detect_metric_anomalies(metric_name, sensitivity)
        
        return {
            "task_type": "anomaly_detection",
            "metric_name": metric_name,
            "sensitivity": sensitivity,
            "anomalies_detected": len(anomalies),
            "anomaly_details": anomalies,
            "monitored_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _monitor_system_health(self, systems: List[str]) -> Dict[str, Any]:
        """Monitor overall system health."""
        health_data = {
            "overall_status": "healthy",
            "systems": {},
            "summary": {
                "healthy_systems": 0,
                "unhealthy_systems": 0,
                "warning_systems": 0
            }
        }
        
        for system in systems:
            system_health = self._get_system_health(system)
            health_data["systems"][system] = system_health
            
            status = system_health["status"]
            if status == "healthy":
                health_data["summary"]["healthy_systems"] += 1
            elif status == "warning":
                health_data["summary"]["warning_systems"] += 1
            else:
                health_data["summary"]["unhealthy_systems"] += 1
        
        # Determine overall status
        if health_data["summary"]["unhealthy_systems"] > 0:
            health_data["overall_status"] = "unhealthy"
        elif health_data["summary"]["warning_systems"] > 0:
            health_data["overall_status"] = "warning"
        
        return health_data
    
    def _track_resource_usage(self, metrics: List[str], duration_hours: float) -> Dict[str, Any]:
        """Track resource usage metrics."""
        resource_data = {
            "tracking_period": f"{duration_hours} hours",
            "metrics": {},
            "summary": {}
        }
        
        for metric in metrics:
            if metric == "cpu":
                cpu_usage = psutil.cpu_percent(interval=1)
                resource_data["metrics"]["cpu"] = {
                    "current": cpu_usage,
                    "average_1h": self._get_average_metric("cpu_usage", 60),
                    "peak_1h": self._get_peak_metric("cpu_usage", 60),
                    "status": "normal" if cpu_usage < 80 else "high"
                }
            elif metric == "memory":
                memory = psutil.virtual_memory()
                resource_data["metrics"]["memory"] = {
                    "current": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3),
                    "status": "normal" if memory.percent < 85 else "high"
                }
            elif metric == "disk":
                disk = psutil.disk_usage('/')
                resource_data["metrics"]["disk"] = {
                    "current": (disk.used / disk.total) * 100,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3),
                    "status": "normal" if (disk.used / disk.total) < 0.9 else "high"
                }
        
        # Store metrics for historical analysis
        timestamp = datetime.now()
        for metric, data in resource_data["metrics"].items():
            self.metrics_history[metric].append({
                "timestamp": timestamp,
                "value": data["current"]
            })
        
        return resource_data
    
    def _analyze_logs(self, log_sources: List[str], analysis_type: str, time_range_hours: int) -> Dict[str, Any]:
        """Analyze system and application logs."""
        analysis_result = {
            "analysis_type": analysis_type,
            "time_range_hours": time_range_hours,
            "log_sources": log_sources,
            "findings": [],
            "statistics": {},
            "recommendations": []
        }
        
        # Mock log analysis based on type
        if analysis_type == "error_detection":
            analysis_result["findings"] = [
                {
                    "level": "ERROR",
                    "count": 15,
                    "message": "Database connection timeout",
                    "first_occurrence": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "last_occurrence": datetime.now().isoformat()
                },
                {
                    "level": "WARNING",
                    "count": 45,
                    "message": "High memory usage detected",
                    "first_occurrence": (datetime.now() - timedelta(hours=6)).isoformat(),
                    "last_occurrence": datetime.now().isoformat()
                }
            ]
            analysis_result["statistics"] = {
                "total_errors": 15,
                "total_warnings": 45,
                "total_info": 1200,
                "error_rate": 0.012
            }
            analysis_result["recommendations"] = [
                "Investigate database connection stability",
                "Consider increasing memory allocation",
                "Review application performance"
            ]
        
        elif analysis_type == "performance_analysis":
            analysis_result["findings"] = [
                {
                    "metric": "response_time",
                    "average": 250,
                    "peak": 2000,
                    "trend": "increasing"
                },
                {
                    "metric": "throughput",
                    "average": 150,
                    "trend": "stable"
                }
            ]
        
        return analysis_result
    
    def _check_service_status(self, service: str, check_type: str) -> Dict[str, Any]:
        """Check the status of a specific service."""
        # Mock service status check
        service_status = {
            "service": service,
            "status": "healthy",  # healthy, unhealthy, degraded
            "response_time": 150,  # ms
            "last_check": datetime.now().isoformat(),
            "uptime": 99.95,  # percentage
            "details": {}
        }
        
        if check_type == "detailed":
            service_status["details"] = {
                "cpu_usage": 25.5,
                "memory_usage": 45.2,
                "connections": 150,
                "error_rate": 0.001
            }
        elif check_type == "connectivity":
            service_status["details"] = {
                "ping_success": True,
                "port_open": True,
                "ssl_valid": True
            }
        
        # Simulate occasional issues
        import random
        if random.random() < 0.1:  # 10% chance of issues
            service_status["status"] = "degraded"
            service_status["response_time"] = 850
        
        return service_status
    
    def _generate_alerts(self, alert_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on monitoring data and rules."""
        new_alerts = []
        
        # Check each alert rule
        for rule_id, rule in self.alert_rules.items():
            if self._evaluate_alert_rule(rule):
                alert = {
                    "id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule_id}",
                    "rule_id": rule_id,
                    "severity": rule["severity"],
                    "title": rule["title"],
                    "message": rule["message"],
                    "timestamp": datetime.now().isoformat(),
                    "status": "active",
                    "metric": rule["metric"],
                    "threshold": rule["threshold"],
                    "current_value": self._get_current_metric_value(rule["metric"])
                }
                
                new_alerts.append(alert)
                self.alerts.append(alert)
        
        return new_alerts
    
    def _create_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring dashboard."""
        dashboard = {
            "dashboard_id": f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": dashboard_config.get("title", "System Monitoring Dashboard"),
            "widgets": [],
            "refresh_interval": dashboard_config.get("refresh_interval", 30),
            "created_at": datetime.now().isoformat()
        }
        
        # Add widgets based on configuration
        widget_types = dashboard_config.get("widgets", ["system_health", "resource_usage", "alerts"])
        
        for widget_type in widget_types:
            if widget_type == "system_health":
                dashboard["widgets"].append({
                    "type": "system_health",
                    "title": "System Health Overview",
                    "data": self._monitor_system_health(["localhost"])
                })
            elif widget_type == "resource_usage":
                dashboard["widgets"].append({
                    "type": "resource_usage",
                    "title": "Resource Usage",
                    "data": self._track_resource_usage(["cpu", "memory", "disk"], 1)
                })
            elif widget_type == "alerts":
                dashboard["widgets"].append({
                    "type": "alerts",
                    "title": "Active Alerts",
                    "data": {
                        "active_alerts": len([a for a in self.alerts if a["status"] == "active"]),
                        "recent_alerts": self.alerts[-5:] if self.alerts else []
                    }
                })
        
        return dashboard
    
    def _analyze_trends(self, metric_name: str, time_range_hours: int = 24) -> Dict[str, Any]:
        """Analyze trends in monitoring metrics."""
        trend_analysis = {
            "metric": metric_name,
            "time_range_hours": time_range_hours,
            "trend_direction": "stable",
            "trend_strength": 0.0,
            "statistics": {},
            "predictions": {}
        }
        
        # Get historical data
        metric_data = list(self.metrics_history.get(metric_name, []))
        
        if len(metric_data) >= 10:
            values = [point["value"] for point in metric_data[-100:]]  # Last 100 points
            
            # Calculate trend
            if len(values) > 1:
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                trend_change = (second_avg - first_avg) / first_avg * 100
                
                if abs(trend_change) > 5:
                    trend_analysis["trend_direction"] = "increasing" if trend_change > 0 else "decreasing"
                    trend_analysis["trend_strength"] = abs(trend_change)
            
            # Calculate statistics
            trend_analysis["statistics"] = {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std_dev": self._calculate_std_dev(values)
            }
            
            # Simple prediction for next hour
            recent_avg = sum(values[-10:]) / len(values[-10:])
            trend_analysis["predictions"] = {
                "next_hour_estimate": recent_avg * (1 + trend_change / 100),
                "confidence": 0.7  # Mock confidence
            }
        
        return trend_analysis
    
    def _monitor_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor network connectivity and performance."""
        network_status = {
            "overall_status": "healthy",
            "connections": {},
            "bandwidth_usage": {},
            "latency_tests": {}
        }
        
        # Mock network monitoring
        endpoints = network_config.get("endpoints", ["google.com", "github.com"])
        
        for endpoint in endpoints:
            # Mock ping test
            latency = 20 + (hash(endpoint) % 50)  # Mock latency 20-70ms
            success_rate = 0.99  # Mock 99% success rate
            
            network_status["latency_tests"][endpoint] = {
                "latency_ms": latency,
                "success_rate": success_rate,
                "status": "healthy" if latency < 100 and success_rate > 0.95 else "degraded"
            }
        
        # Mock bandwidth usage
        network_status["bandwidth_usage"] = {
            "upload_mbps": 45.2,
            "download_mbps": 98.7,
            "utilization_percent": 35.5
        }
        
        return network_status
    
    async def _continuous_monitoring(self, interval_seconds: int):
        """Run continuous monitoring in the background."""
        while self.monitoring_active:
            try:
                # Collect metrics
                self._track_resource_usage(["cpu", "memory", "disk"], 0.1)
                
                # Check for alerts
                self._generate_alerts({})
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(interval_seconds)
    
    # Helper methods
    def _setup_default_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Setup default alert rules."""
        return {
            "high_cpu": {
                "metric": "cpu_usage",
                "threshold": 80,
                "operator": ">",
                "severity": "warning",
                "title": "High CPU Usage",
                "message": "CPU usage is above 80%"
            },
            "high_memory": {
                "metric": "memory_usage",
                "threshold": 85,
                "operator": ">",
                "severity": "warning",
                "title": "High Memory Usage",
                "message": "Memory usage is above 85%"
            },
            "low_disk": {
                "metric": "disk_usage",
                "threshold": 90,
                "operator": ">",
                "severity": "critical",
                "title": "Low Disk Space",
                "message": "Disk usage is above 90%"
            }
        }
    
    def _evaluate_alert_rule(self, rule: Dict[str, Any]) -> bool:
        """Evaluate if an alert rule condition is met."""
        current_value = self._get_current_metric_value(rule["metric"])
        threshold = rule["threshold"]
        operator = rule["operator"]
        
        if operator == ">":
            return current_value > threshold
        elif operator == "<":
            return current_value < threshold
        elif operator == "==":
            return current_value == threshold
        
        return False
    
    def _get_current_metric_value(self, metric_name: str) -> float:
        """Get current value for a metric."""
        if metric_name == "cpu_usage":
            return psutil.cpu_percent()
        elif metric_name == "memory_usage":
            return psutil.virtual_memory().percent
        elif metric_name == "disk_usage":
            return (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        
        # Return mock value for unknown metrics
        return 50.0
    
    def _get_system_health(self, system: str) -> Dict[str, Any]:
        """Get health status for a system."""
        # Mock system health check
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        
        # Determine overall status
        if cpu_usage > 90 or memory_usage > 95 or disk_usage > 95:
            status = "unhealthy"
        elif cpu_usage > 80 or memory_usage > 85 or disk_usage > 90:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "system": system,
            "status": status,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "uptime": self._get_system_uptime(),
            "last_check": datetime.now().isoformat()
        }
    
    def _get_resource_metrics(self, resource_type: str) -> Dict[str, Any]:
        """Get metrics for a specific resource type."""
        if resource_type == "cpu":
            return {
                "usage_percent": psutil.cpu_percent(interval=1),
                "core_count": psutil.cpu_count(),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
        elif resource_type == "memory":
            memory = psutil.virtual_memory()
            return {
                "usage_percent": memory.percent,
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3)
            }
        elif resource_type == "disk":
            disk = psutil.disk_usage('/')
            return {
                "usage_percent": (disk.used / disk.total) * 100,
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3)
            }
        elif resource_type == "network":
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        
        return {}
    
    def _calculate_uptime(self, service: str) -> Dict[str, Any]:
        """Calculate uptime for a service."""
        # Mock uptime calculation
        uptime_hours = 720  # 30 days
        total_checks = 8640  # Every 5 minutes for 30 days
        successful_checks = 8580
        
        uptime_percentage = (successful_checks / total_checks) * 100
        
        return {
            "service": service,
            "uptime_percentage": uptime_percentage,
            "uptime_hours": uptime_hours,
            "total_checks": total_checks,
            "successful_checks": successful_checks,
            "failed_checks": total_checks - successful_checks,
            "last_downtime": (datetime.now() - timedelta(hours=48)).isoformat()
        }
    
    def _detect_metric_anomalies(self, metric_name: str, sensitivity: str) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data."""
        anomalies = []
        metric_data = list(self.metrics_history.get(metric_name, []))
        
        if len(metric_data) < 20:  # Need sufficient data
            return anomalies
        
        values = [point["value"] for point in metric_data]
        mean_val = sum(values) / len(values)
        std_dev = self._calculate_std_dev(values)
        
        # Set threshold based on sensitivity
        threshold_multiplier = {'low': 3, 'medium': 2, 'high': 1.5}[sensitivity]
        threshold = threshold_multiplier * std_dev
        
        # Check recent values for anomalies
        recent_data = metric_data[-10:]  # Last 10 data points
        
        for point in recent_data:
            deviation = abs(point["value"] - mean_val)
            if deviation > threshold:
                anomalies.append({
                    "timestamp": point["timestamp"].isoformat(),
                    "value": point["value"],
                    "expected_range": [mean_val - threshold, mean_val + threshold],
                    "deviation": deviation,
                    "severity": "high" if deviation > 2 * threshold else "medium"
                })
        
        return anomalies
    
    def _get_average_metric(self, metric_name: str, minutes_back: int) -> float:
        """Get average value for a metric over the specified time period."""
        metric_data = list(self.metrics_history.get(metric_name, []))
        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
        
        recent_values = [
            point["value"] for point in metric_data
            if point["timestamp"] >= cutoff_time
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else 0.0
    
    def _get_peak_metric(self, metric_name: str, minutes_back: int) -> float:
        """Get peak value for a metric over the specified time period."""
        metric_data = list(self.metrics_history.get(metric_name, []))
        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
        
        recent_values = [
            point["value"] for point in metric_data
            if point["timestamp"] >= cutoff_time
        ]
        
        return max(recent_values) if recent_values else 0.0
    
    def _get_system_uptime(self) -> float:
        """Get system uptime in hours."""
        import time
        uptime_seconds = time.time() - psutil.boot_time()
        return uptime_seconds / 3600  # Convert to hours
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
