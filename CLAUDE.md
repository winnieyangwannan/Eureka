# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eureka is a reinforcement learning research project that uses Large Language Models (LLMs) to automatically generate reward functions for complex robotic manipulation tasks. The system uses GPT models to iteratively design and refine reward functions that are then trained using RL algorithms in Isaac Gym environments.

## Architecture

The project consists of three main components:

### 1. Core Eureka System (`eureka/`)
- **Main script**: `eureka/eureka.py` - Orchestrates the LLM-RL feedback loop
- **Environment definitions**: `eureka/envs/` - Contains task definitions split into `isaac/` (Isaac Gym) and `bidex/` (dexterous hand tasks)
- **Utilities**: `eureka/utils/` - Helper functions for file processing, task creation, and environment pruning
- **Configuration**: `eureka/cfg/` - Hydra configs for environments and main parameters

### 2. Isaac Gym Integration (`isaacgymenvs/`)
- Modified Isaac Gym environments that accept Eureka-generated reward functions
- RL training infrastructure based on rl_games
- Task implementations that can be dynamically modified with new reward functions

### 3. RL Training Backend (`rl_games/`)
- Modified rl_games library for policy training
- Supports the specific requirements of Eureka's reward injection system

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n eureka python=3.8
conda activate eureka

# Install Isaac Gym (external dependency required)
# Follow Isaac Gym installation guide first, then:
cd isaacgym/python
pip install -e .

# Install Eureka
cd Eureka
pip install -e .
cd isaacgymenvs && pip install -e .
cd ../rl_games && pip install -e .
```

### Running Eureka
```bash
# Basic run with environment and iteration settings
python eureka.py env={environment} iteration={num_iterations} sample={num_samples}

# Example commands
python eureka.py env=shadow_hand sample=4 iteration=2 model=gpt-4-0314
python eureka.py env=humanoid sample=16 iteration=5 model=gpt-3.5-turbo-16k-0613
```

### Testing Environments
```bash
# Test Isaac Gym environment directly
cd isaacgymenvs/isaacgymenvs
python train.py task=YOUR_TASK

# Visualize trained policy (e.g., pen spinning demo)
python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth
```

### Adding New Environments
```bash
# Create pruned environment code for Eureka context
cd eureka/utils
python prune_env.py your_new_task
```

## Key Configuration Files

- `eureka/cfg/config.yaml` - Main configuration including model settings, iteration counts, and evaluation parameters
- `eureka/cfg/env/*.yaml` - Environment-specific configurations with task names and descriptions
- Environment configs support these key parameters:
  - `env_name`: Internal environment identifier
  - `task`: Isaac Gym task name
  - `description`: Natural language task description for LLM

## Important Implementation Details

### Reward Function Integration
- Eureka generates reward functions that are dynamically injected into Isaac Gym environments
- The system expects environments to have either `def compute_reward(self):` or `def compute_reward(self, actions):` methods
- Generated rewards return both `self.rew_buf` and `self.rew_dict` for detailed logging

### LLM Integration
- Requires OpenAI API key set as `OPENAI_API_KEY` environment variable
- Supports multiple OpenAI models (GPT-4, GPT-3.5-turbo variants)
- Uses iterative feedback loop where RL training results inform subsequent LLM generations

### File Structure for Generated Content
- Each Eureka run creates timestamped output directory with:
  - Generated reward functions (`env_iter*_response*.py`)
  - RL training logs (`env_iter*_response*.txt`)
  - Summary plots and evaluation results
  - Message history for LLM conversation (`messages.json`)

## Dependencies

Critical dependencies with specific versions:
- `torch<=2.0.0`
- `numpy==1.20.0`
- `gym==0.23.1`
- Isaac Gym (external installation required)
- OpenAI API access for LLM queries

The system is designed for Ubuntu 20.04/22.04 with NVIDIA GPUs for Isaac Gym simulation.