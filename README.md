# kanboard_huggingface_AI-Agent

# Kanboard Hugging Face AI Agent

A Python agent that fetches tasks from a Kanboard project, classifies them (WIP / Blocked), generates a summary using Hugging Face LLMs, and optionally sends the report via email.

## Features

- Fetch tasks from Kanboard via JSON-RPC API.
- Classify tasks into **Work In Progress** and **Blocked**.
- Generate AI summaries using Hugging Face models (LLM).
- Save reports locally.
- Optionally send email reports.

## Requirements

- Python 3.10+
- Hugging Face API key
- Kanboard API token
- Email account (SMTP)

## Installation

```bash
git clone https://github.com/YourUsername/kanboard-huggingface-agent.git
cd kanboard-huggingface-agent
pip install -r requirements.txt
