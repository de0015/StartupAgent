"""
Core configuration and shared utilities for the multi-agent system.
"""

import os
from typing import Dict, Any
from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # Ollama Configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama2")
    
    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./multi_agent.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    
    # Performance Monitoring
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    
    # Agent Configuration
    max_concurrent_agents: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "10"))
    agent_timeout: int = int(os.getenv("AGENT_TIMEOUT", "300"))
    retry_attempts: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    
    # Dashboard Configuration
    dashboard_refresh_interval: int = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "5"))
    enable_real_time_updates: bool = os.getenv("ENABLE_REAL_TIME_UPDATES", "true").lower() == "true"

# Global settings instance
settings = Settings()

# Agent Types
AGENT_TYPES = {
    "customer_service": "Customer Service Agent",
    "data_analysis": "Data Analysis Agent", 
    "workflow_automation": "Workflow Automation Agent",
    "integration": "Integration Agent",
    "monitoring": "Monitoring Agent"
}

# Task Priorities
TASK_PRIORITIES = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4
}

# Agent States
AGENT_STATES = {
    "idle": "idle",
    "busy": "busy",
    "error": "error",
    "offline": "offline"
}
