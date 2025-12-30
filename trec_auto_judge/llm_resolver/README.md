# Participant-requested LLM Model Configuration

This guide explains how to request specific LLM models for your AutoJudge implementation.

## Quick Start

Create a file named `llm-config.yml` in your judge directory:

```yaml
model_preferences:
  - "gpt-4o"
  - "gpt-4-turbo"
  - "claude-3-5-sonnet"
```

For development you pass it to your judge via the `--llm-config` flag:

```bash
./my-judge.py --rag-responses runs/ --llm-config llm-config.yml --output output.txt
```

## How It Works

1. You declare an **ordered list** of preferred models in `llm-config.yml`
2. At runtime, the system checks which models are available from the organizer's pool
3. The **first available** model from your list is selected
4. Your judge receives a ready-to-use `MinimaLlmConfig` with the resolved model

## Configuration Format

### Basic Configuration

```yaml
model_preferences:
  - "gpt-4o"           # First choice
  - "gpt-4-turbo"      # Fallback if gpt-4o unavailable
  - "llama-3.1-70b"    # Second fallback
```

### With Fallback Behavior

```yaml
model_preferences:
  - "gpt-4o"
  - "claude-3-5-sonnet"

# What happens if none of your preferences are available?
# Options: "error" (default) or "use_default"
on_no_match: "error"
```

- `on_no_match: "error"` - Fail with an error message listing available models
- `on_no_match: "use_default"` - Fall back to the organizer's default model

## Available Models

The available models are controlled by the evaluation organizers. Common models include:

| Model Name | Description |
|------------|-------------|
| `gpt-4o` | OpenAI GPT-4o |
| `gpt-4o-mini` | OpenAI GPT-4o Mini (faster, cheaper) |
| `llama-3.1-70b` | Meta Llama 3.1 70B |

**Note:** Actual availability depends on the evaluation environment. If your preferred model is unavailable, the system will try your fallback choices.

## Using the Resolved Config in Your Judge

Your judge receives the resolved `MinimaLlmConfig` as the third parameter:

```python
from trec_auto_judge import AutoJudge
from trec_auto_judge.llm import MinimaLlmConfig, OpenAIMinimaLlm

class MyJudge(AutoJudge):
    def judge(self, rag_responses, rag_topics, llm_config: MinimaLlmConfig):
        # Create LLM backend from resolved config
        llm = OpenAIMinimaLlm(llm_config)

        # Use llm for your judging logic...
        # llm_config.model contains the resolved model name
        # llm_config.base_url contains the endpoint URL
```

## Fallback: Environment Variables

If no `--llm-config` is provided, the system falls back to environment variables:

- `OPENAI_BASE_URL` - LLM endpoint URL
- `OPENAI_MODEL` - Model identifier
- `OPENAI_API_KEY` - API key/token

This ensures backwards compatibility with existing judges.

## Example Directory Structure

```
my-judge/
├── my-judge.py           # Your judge implementation
├── llm-config.yml        # Model preferences
├── requirements.txt
└── Dockerfile
```

## Troubleshooting

### "No model available from preferences"

Your preferred models are not in the available pool. Either:
1. Add more fallback options to your `model_preferences` list
2. Set `on_no_match: "use_default"` to accept the organizer's default
3. Check the error message for the list of available models

### Config not being read

Ensure you're passing the `--llm-config` flag:
```bash
./my-judge.py --llm-config llm-config.yml ...
```

### YAML parsing errors

Verify your YAML syntax. Common issues:
- Use spaces, not tabs
- List items need `- ` prefix with space
- Strings with special characters need quotes