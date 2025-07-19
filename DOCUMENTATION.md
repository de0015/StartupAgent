# Multi-Agent System Documentation

## Overview

This is a comprehensive enterprise-grade multi-agent framework built with LangChain and Ollama LLM. The system provides automated business process management through specialized AI agents working in coordination.

## Architecture

### Core Components

1. **Agent Manager** (`core/agent_manager.py`)
   - Central coordination and task distribution
   - Priority queue management
   - Inter-agent communication
   - Health monitoring

2. **Performance Monitor** (`core/performance_monitor.py`)
   - Real-time metrics collection
   - Performance analytics
   - Anomaly detection
   - ROI tracking

3. **Communication System** (`core/communication.py`)
   - Message broker for agent communication
   - WebSocket support for real-time updates
   - Event-driven architecture

4. **Web Dashboard** (`dashboard/dashboard.py`)
   - Real-time monitoring interface
   - Task management
   - Performance visualization
   - RESTful API endpoints

### Agent Types

#### 1. Customer Service Agents (`agents/customer_service/`)
- **Capabilities:**
  - Handle customer inquiries
  - Appointment scheduling
  - Order status checking
  - Complaint handling
  - Support ticket creation

- **Sample Tasks:**
  ```json
  {
    "type": "customer_inquiry",
    "data": {
      "customer_id": "CUST001",
      "message": "I need help with my order",
      "priority": "high"
    }
  }
  ```

#### 2. Data Analysis Agents (`agents/data_analysis/`)
- **Capabilities:**
  - Sales analysis and reporting
  - Customer behavior analysis
  - Trend identification
  - Data visualization
  - Predictive modeling

- **Sample Tasks:**
  ```json
  {
    "type": "sales_analysis",
    "data": {
      "period": "monthly",
      "metrics": ["revenue", "conversion_rate"]
    }
  }
  ```

#### 3. Workflow Automation Agents (`agents/workflow_automation/`)
- **Capabilities:**
  - Document processing
  - Data transformation
  - File organization
  - Automated reporting
  - Batch processing

- **Sample Tasks:**
  ```json
  {
    "type": "document_processing",
    "data": {
      "document_path": "/path/to/document.pdf",
      "processing_type": "extract_text"
    }
  }
  ```

#### 4. Integration Agents (`agents/integration/`)
- **Capabilities:**
  - API integrations
  - Database synchronization
  - Data migration
  - Webhook handling
  - ETL operations

- **Sample Tasks:**
  ```json
  {
    "type": "api_integration",
    "data": {
      "api_endpoint": "https://api.example.com",
      "integration_type": "rest"
    }
  }
  ```

#### 5. Monitoring Agents (`agents/monitoring/`)
- **Capabilities:**
  - System health monitoring
  - Performance tracking
  - Alert management
  - Log analysis
  - Resource monitoring

- **Sample Tasks:**
  ```json
  {
    "type": "system_monitoring",
    "data": {
      "systems": ["web_server", "database"],
      "interval_seconds": 60
    }
  }
  ```

## Installation & Setup

### Prerequisites

1. **Python 3.8+**
2. **Ollama** - Download from [https://ollama.ai/](https://ollama.ai/)
3. **Git** (optional)

### Quick Start

1. **Install Ollama and pull a model:**
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull llama2  # or llama3, mistral, etc.
   ollama serve        # Start Ollama service
   ```

2. **Run the startup script:**
   ```powershell
   # Navigate to the project directory
   cd multi_agent_system
   
   # Run the startup script (Windows PowerShell)
   .\start.ps1
   ```

3. **Access the dashboard:**
   - Open your browser to `http://localhost:8000`

### Manual Installation

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Start the system:**
   ```bash
   python main.py
   ```

## Configuration

### Environment Variables (`.env`)

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Agent Configuration
MAX_CONCURRENT_AGENTS=10
AGENT_TIMEOUT=300

# Dashboard Configuration
DASHBOARD_REFRESH_INTERVAL=5
ENABLE_REAL_TIME_UPDATES=true
```

### Agent Configuration

Each agent can be configured with:
- **Capabilities**: List of task types the agent can handle
- **Performance settings**: Timeouts, retry attempts
- **LLM parameters**: Temperature, model selection
- **Tool configurations**: API keys, database connections

## Usage

### Web Dashboard

The web dashboard provides:

1. **System Overview** - Real-time metrics and status
2. **Agent Management** - Monitor and control individual agents
3. **Task Queue** - View pending, active, and completed tasks
4. **Performance Analytics** - Success rates, throughput, response times
5. **Real-time Updates** - WebSocket-based live data

### API Endpoints

#### Submit a Task
```http
POST /api/tasks
Content-Type: application/json

{
  "type": "customer_inquiry",
  "data": {
    "customer_id": "CUST001",
    "message": "Need help with order"
  },
  "priority": "high"
}
```

#### Get Task Status
```http
GET /api/tasks/{task_id}
```

#### Get System Status
```http
GET /api/system/status
```

#### Get Agent Status
```http
GET /api/agents/{agent_id}
```

### Demo Script

Run the interactive demo to explore system capabilities:

```powershell
.\demo.ps1
```

The demo provides:
- Sample task submissions
- Real-time monitoring
- Performance analytics
- System status checks

## Development

### Adding New Agent Types

1. **Create agent class:**
   ```python
   from core.base_agent import BaseAgent, Task
   
   class MyCustomAgent(BaseAgent):
       def __init__(self, agent_id: str):
           capabilities = ["my_task_type"]
           super().__init__(agent_id, "my_agent", capabilities)
       
       async def setup_agent(self):
           # Setup tools and configurations
           pass
       
       async def process_task(self, task: Task):
           # Process the task
           return {"result": "success"}
   ```

2. **Register in main.py:**
   ```python
   agent = MyCustomAgent("my_agent_1")
   await agent.setup_agent()
   self.agent_manager.register_agent(agent)
   ```

### Adding New Task Types

1. **Define task handling in appropriate agent**
2. **Add to agent capabilities list**
3. **Update API documentation**

### Custom Tools

Agents can use custom tools:

```python
from langchain.tools import Tool

def my_custom_function(input_data):
    # Your custom logic
    return result

tool = Tool(
    name="my_tool",
    description="Description of what the tool does",
    func=my_custom_function
)
```

## Monitoring & Maintenance

### Performance Metrics

The system tracks:
- **Task throughput** - Tasks processed per minute
- **Success rates** - Percentage of successfully completed tasks
- **Response times** - Average task completion duration
- **Agent utilization** - Active vs idle time
- **System resources** - CPU, memory, disk usage

### Health Checks

Automated health monitoring includes:
- Agent responsiveness
- LLM connectivity
- Database connections
- API endpoint availability
- Resource usage thresholds

### Alerts & Notifications

Configure alerts for:
- High error rates
- Performance degradation
- Resource exhaustion
- Agent failures
- Task queue backlog

### Logging

Structured logging with:
- **Level**: DEBUG, INFO, WARNING, ERROR
- **Format**: JSON for machine parsing
- **Retention**: Configurable log rotation
- **Monitoring**: Integration with log analysis tools

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```
   Solution: Ensure Ollama is running on localhost:11434
   Check: curl http://localhost:11434/api/version
   ```

2. **Agent Not Responding**
   ```
   Solution: Check agent health in dashboard
   Restart: Individual agents can be restarted via API
   ```

3. **High Memory Usage**
   ```
   Solution: Adjust MAX_CONCURRENT_AGENTS
   Monitor: Check agent task completion rates
   ```

4. **Task Queue Backlog**
   ```
   Solution: Scale up agents or optimize task processing
   Monitor: Dashboard shows queue depth and processing rates
   ```

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
```

### Performance Tuning

- **Agent Count**: Scale based on workload
- **Task Priorities**: Use priority queuing effectively
- **Resource Limits**: Set appropriate timeouts and limits
- **Caching**: Implement caching for repeated operations

## Security Considerations

1. **API Authentication**: Implement authentication for production
2. **Input Validation**: Validate all task inputs
3. **Rate Limiting**: Prevent abuse of API endpoints
4. **Data Privacy**: Ensure sensitive data handling compliance
5. **Network Security**: Use HTTPS in production

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: multi-agent-system
  template:
    metadata:
      labels:
        app: multi-agent-system
    spec:
      containers:
      - name: multi-agent-system
        image: multi-agent-system:latest
        ports:
        - containerPort: 8000
```

### Scaling Recommendations

- **Horizontal scaling**: Multiple instances behind load balancer
- **Database**: Use external database for persistence
- **Message Queue**: Redis or RabbitMQ for high throughput
- **Monitoring**: Prometheus + Grafana for production monitoring

## Support & Contributing

### Getting Help

1. Check the troubleshooting section
2. Review logs for error messages
3. Check system status in dashboard
4. Consult API documentation

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

### License

MIT License - see LICENSE file for details.

## Changelog

### Version 1.0.0
- Initial release with core agent types
- Web dashboard implementation
- Real-time monitoring
- Performance analytics
- RESTful API
- Demo system

---

For technical support or questions, please refer to the project documentation or create an issue in the repository.
