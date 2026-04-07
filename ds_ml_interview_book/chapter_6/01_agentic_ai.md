# Chapter 6: Agentic AI & DevOps

## AI Agents Architecture

### Q1: What are AI agents? How do they differ from traditional APIs?

**A:**

```python
from anthropic import Anthropic

class AIAgent:
    """
    AI Agent = LLM + Tools + Memory + Planning
    
    Components:
    1. LLM (reasoning engine)
    2. Tools (actions the agent can take)
    3. Memory (conversation history, context)
    4. Planning (breaking down complex tasks)
    """
    
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = []
        self.tools = self._define_tools()
    
    def _define_tools(self):
        return [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "search_web",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def execute_tool(self, tool_name, tool_input):
        """Execute a tool and return results"""
        if tool_name == "get_weather":
            return self._get_weather(tool_input["location"])
        elif tool_name == "search_web":
            return self._search_web(tool_input["query"])
        else:
            return "Tool not found"
    
    def _get_weather(self, location):
        # Mock implementation
        return f"Weather in {location}: 72°F, sunny"
    
    def _search_web(self, query):
        # Mock implementation
        return f"Search results for '{query}': [result1, result2, result3]"
    
    def run(self, user_message):
        """Main agent loop with tool use"""
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Agent loop: LLM decides which tools to use
        while True:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=self.tools,
                messages=self.conversation_history
            )
            
            # Check if agent wants to use a tool
            if response.stop_reason == "tool_use":
                tool_use_block = next(
                    block for block in response.content
                    if block.type == "tool_use"
                )
                
                tool_name = tool_use_block.name
                tool_input = tool_use_block.input
                
                print(f"Agent using tool: {tool_name} with input {tool_input}")
                
                # Execute the tool
                tool_result = self.execute_tool(tool_name, tool_input)
                
                # Add tool use and result to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                self.conversation_history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": tool_result
                        }
                    ]
                })
                
                # Continue the loop to let agent use more tools or respond
            else:
                # Agent is done, return final response
                final_text = next(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
                return final_text

# Usage
agent = AIAgent(api_key="your-api-key")
response = agent.run("What's the weather in San Francisco and can you search for recent news about AI?")
print(response)
```

**Traditional API vs AI Agent:**

| Traditional API | AI Agent |
|---|---|
| Fixed input/output | Dynamic reasoning |
| Single function call | Multi-step planning |
| No memory | Conversation history |
| Deterministic | Probabilistic |
| Code defines logic | LLM decides logic |

---

### Q2: Build a multi-agent system for data science workflow automation

**A:**

```python
from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic

# Initialize LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Agent 1: Data Analyst
data_analyst = Agent(
    role="Data Analyst",
    goal="Analyze datasets and identify patterns",
    backstory="Expert in exploratory data analysis and statistical inference",
    llm=llm,
    tools=[],  # Add data analysis tools
    verbose=True
)

# Agent 2: Feature Engineer
feature_engineer = Agent(
    role="Feature Engineer",
    goal="Create predictive features from raw data",
    backstory="Specialist in feature engineering and dimensionality reduction",
    llm=llm,
    verbose=True
)

# Agent 3: Model Trainer
model_trainer = Agent(
    role="ML Engineer",
    goal="Train and tune machine learning models",
    backstory="Expert in model selection, hyperparameter tuning, and evaluation",
    llm=llm,
    verbose=True
)

# Agent 4: Deployment Specialist
deployment_agent = Agent(
    role="MLOps Engineer",
    goal="Deploy models to production and monitor performance",
    backstory="Specialist in model serving, monitoring, and CI/CD",
    llm=llm,
    verbose=True
)

# Define tasks
task1 = Task(
    description="Analyze the customer churn dataset. Identify key patterns and correlations.",
    agent=data_analyst,
    expected_output="EDA report with visualizations and statistical insights"
)

task2 = Task(
    description="Based on the EDA, engineer features that predict customer churn.",
    agent=feature_engineer,
    expected_output="Feature engineering pipeline with new features and importance scores"
)

task3 = Task(
    description="Train multiple models and select the best performer using cross-validation.",
    agent=model_trainer,
    expected_output="Trained model with evaluation metrics and comparison table"
)

task4 = Task(
    description="Deploy the best model to production with monitoring and A/B testing.",
    agent=deployment_agent,
    expected_output="Deployment plan with monitoring dashboards and rollback strategy"
)

# Create crew
crew = Crew(
    agents=[data_analyst, feature_engineer, model_trainer, deployment_agent],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,  # Tasks run in order
    verbose=2
)

# Execute
result = crew.kickoff()
print(result)
```

**Key concepts:**
- **Agents**: Specialized AI entities with specific roles
- **Tasks**: Clear objectives with expected outputs
- **Tools**: Functions agents can call (APIs, databases, file systems)
- **Memory**: Shared context across agents
- **Process**: Sequential, hierarchical, or concurrent execution

---

### Q3: How do you handle agent failures and retries in production?

**A:**

```python
import time
from functools import wraps
from typing import Callable, Any

class AgentRetryStrategy:
    """Production-grade retry logic for AI agents"""
    
    def __init__(
        self,
        max_retries=3,
        backoff_factor=2,
        retry_exceptions=(Exception,),
        fallback_response="I encountered an error. Please try again."
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_exceptions = retry_exceptions
        self.fallback_response = fallback_response
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                
                except self.retry_exceptions as e:
                    last_exception = e
                    wait_time = self.backoff_factor ** attempt
                    
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {wait_time}s...")
                    
                    time.sleep(wait_time)
            
            # All retries exhausted
            print(f"All retries exhausted. Last error: {last_exception}")
            return self.fallback_response
        
        return wrapper

# Usage
@AgentRetryStrategy(max_retries=3, backoff_factor=2)
def query_agent(prompt):
    # Simulated agent call that might fail
    from anthropic import Anthropic
    client = Anthropic(api_key="your-key")
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Production pattern: Circuit breaker
class CircuitBreaker:
    """Prevent cascading failures"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            print("Circuit breaker opened!")
```

---

## Agentic DevOps

### Q4: How would you build a CI/CD pipeline with AI agents?

**A:**

```yaml
# .github/workflows/ai-agent-cicd.yml
name: AI Agent CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: AI Code Review
        run: |
          python scripts/ai_code_review.py \
            --files-changed $(git diff --name-only HEAD~1) \
            --model claude-3-5-sonnet-20241022
      
      - name: Post Review Comments
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const review = require('./review_output.json');
            github.rest.pulls.createReview({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: context.issue.number,
              body: review.summary,
              event: 'COMMENT'
            });
  
  test-generation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Generate Tests with AI
        run: |
          python scripts/generate_tests.py \
            --coverage-threshold 80 \
            --model claude-3-5-sonnet-20241022
      
      - name: Run Generated Tests
        run: pytest tests/
  
  deployment-agent:
    needs: [code-review, test-generation]
    runs-on: ubuntu-latest
    steps:
      - name: AI Deployment Planner
        run: |
          python scripts/deployment_agent.py \
            --environment production \
            --canary-percentage 10
      
      - name: Execute Deployment
        run: |
          kubectl apply -f deployment.yaml
```

**AI Agent Scripts:**

```python
# scripts/ai_code_review.py
from anthropic import Anthropic
import subprocess
import json

def get_git_diff():
    result = subprocess.run(
        ['git', 'diff', 'HEAD~1'],
        capture_output=True,
        text=True
    )
    return result.stdout

def ai_code_review(diff_content):
    client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    
    prompt = f"""
Review this code diff for:
1. Bugs and logic errors
2. Security vulnerabilities
3. Performance issues
4. Code style and best practices
5. Missing tests

Diff:
```
{diff_content}
```

Provide:
- Summary (2-3 sentences)
- Issues (list with severity: high/medium/low)
- Suggestions (actionable fixes)
"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    
    review = response.content[0].text
    
    # Save for GitHub Action
    with open('review_output.json', 'w') as f:
        json.dump({"summary": review}, f)
    
    return review

if __name__ == "__main__":
    diff = get_git_diff()
    review = ai_code_review(diff)
    print(review)
```

---

### Q5: Build an AI agent for incident response and root cause analysis

**A:**

```python
from anthropic import Anthropic
from dataclasses import dataclass
from typing import List
import time

@dataclass
class Incident:
    id: str
    title: str
    severity: str
    logs: List[str]
    metrics: dict
    timestamp: float

class IncidentResponseAgent:
    """AI agent for automated incident triage and RCA"""
    
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.tools = self._define_tools()
    
    def _define_tools(self):
        return [
            {
                "name": "query_logs",
                "description": "Query application logs from Elasticsearch",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "time_range": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_metrics",
                "description": "Fetch metrics from Prometheus/Datadog",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string"},
                        "time_range": {"type": "string"}
                    },
                    "required": ["metric_name"]
                }
            },
            {
                "name": "check_recent_deployments",
                "description": "Check recent code deployments",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"}
                    },
                    "required": ["service"]
                }
            },
            {
                "name": "create_ticket",
                "description": "Create a Jira ticket for the incident",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "assignee": {"type": "string"}
                    },
                    "required": ["title", "description"]
                }
            }
        ]
    
    def analyze_incident(self, incident: Incident):
        """Perform root cause analysis"""
        
        analysis_prompt = f"""
You are an SRE investigating a production incident.

Incident: {incident.title}
Severity: {incident.severity}
Recent logs: {incident.logs[:10]}
Metrics: {incident.metrics}

Your tasks:
1. Identify the root cause
2. Determine if recent deployments are related
3. Suggest immediate mitigation steps
4. Recommend long-term fixes

Use the available tools to gather more information.
"""
        
        conversation = [{"role": "user", "content": analysis_prompt}]
        
        # Agent loop
        while True:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                tools=self.tools,
                messages=conversation
            )
            
            if response.stop_reason == "tool_use":
                # Agent wants to use a tool
                for block in response.content:
                    if block.type == "tool_use":
                        tool_result = self._execute_tool(
                            block.name,
                            block.input
                        )
                        
                        conversation.append({
                            "role": "assistant",
                            "content": response.content
                        })
                        
                        conversation.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_result
                            }]
                        })
            else:
                # Agent is done
                return self._extract_rca(response.content[0].text)
    
    def _execute_tool(self, tool_name, tool_input):
        """Execute incident response tools"""
        if tool_name == "query_logs":
            # Simulate log query
            return "Error logs show 500 errors spiking at 14:32 UTC"
        
        elif tool_name == "get_metrics":
            # Simulate metrics fetch
            return "CPU usage normal, but database connections maxed out"
        
        elif tool_name == "check_recent_deployments":
            # Simulate deployment check
            return "Deployment v2.3.1 rolled out at 14:30 UTC"
        
        elif tool_name == "create_ticket":
            # Simulate ticket creation
            return f"Ticket INC-{int(time.time())} created"
    
    def _extract_rca(self, agent_response):
        """Parse agent's root cause analysis"""
        return {
            "rca": agent_response,
            "timestamp": time.time()
        }

# Usage
agent = IncidentResponseAgent(api_key="your-key")

incident = Incident(
    id="INC-12345",
    title="API 500 errors spike",
    severity="high",
    logs=["ERROR: Database connection timeout", "ERROR: 500 Internal Server Error"],
    metrics={"cpu": 45, "memory": 78, "db_connections": 100},
    timestamp=time.time()
)

rca = agent.analyze_incident(incident)
print(rca)
```

---

## Summary: Agentic AI Checklist

| Concept | Key Insight | Interview Signal |
|---|---|---|
| AI Agents | LLM + Tools + Memory + Planning | "Agents reason about which tools to use, unlike fixed APIs" |
| Multi-agent systems | Specialized agents collaborate on tasks | "CrewAI for orchestration, agents for specialized roles" |
| Retry/fallback | Handle LLM failures gracefully | "Exponential backoff + circuit breaker pattern" |
| CI/CD agents | Automate code review, test generation, deployment | "AI agents review PRs, generate tests, plan deployments" |
| Incident response | Automated RCA with tool use | "Agent queries logs, metrics, deployments to find root cause" |
