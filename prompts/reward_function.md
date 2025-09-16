Tell me more about Reward Function and LLM Integration.

Overall Questions: 
1. **Directory Structure Overview**: List all files and folders related to reward function integration with their purposes
2. **High-Level Architecture**: Explain how the components work together as a system
3. **Key Functions Analysis**: Walk through the most important functions and their execution flow
4. **Dependencies and Relationships**: Show how files import/reference each other
5. **Design Patterns**: Identify any architectural patterns used 

Specific Questions:
1. How does Eureka generates reward functions that are dynamically injected into Isaac Gym environments.

2. Tell me more about `def compute_reward(self)` and `def compute_reward(self, actions)` methods. 

3. Generated rewards return both `self.rew_buf` and `self.rew_dict` for detailed logging. What are rew_buf and raw_dict?

4. How does the code implement iterative feedback loop where RL training results inform subsequent LLM generations? What are the inputs provided to LLM and what are the outputs from LLM?


---
## LLM-RL Pipeline (eureka/):
  - eureka/eureka.py - Main orchestration loop coordinating LLM
  queries and RL training
  - eureka/utils/extract_task_code.py - Function signature parsing and
   code injection utilities
  - eureka/utils/misc.py - GPU management and training monitoring
  utilities
  - eureka/utils/file_utils.py - TensorBoard log processing and
  dynamic code loading

## Prompt Engineering (eureka/utils/prompts/):
  - initial_system.txt - LLM system instructions for reward
  engineering
  - initial_user.txt - Task context and environment description
  template
  - reward_signature.txt - Function signature template for generated
  rewards
  - policy_feedback.txt - Training results feedback template
  - code_feedback.txt - Iterative improvement instructions
  - execution_error_feedback.txt - Error handling and debugging
  promp


---
## Main Execution Flow (eureka.py:22-397):

### Phase 1: Initialization
  1. Load configuration - Task description, LLM model, iteration
  parameters
  2. Prepare prompts - Combine system instructions with environment
  context
  3. Initialize message history - Start LLM conversation thread



### Phase 2: Generation Loop (per iteration)
  1. LLM Query (eureka.py:91-113) - Request multiple reward function
  samples
  2. Code Extraction (eureka.py:126-146) - Parse Python code from LLM
  responses using regex patterns
  3. Function Injection (eureka.py:147-178) - Dynamically inject
  reward functions into environment files
  4. RL Training Launch (eureka.py:186-200) - Spawn parallel training
  processes
  5. Results Collection (eureka.py:210-277) - Gather TensorBoard logs
  and performance metrics


### phase 3: Feedback & Selection
  1. Performance Analysis (eureka.py:288-310) - Rank samples by
  success rate and reward correlation
  2. Feedback Generation - Create detailed training summaries for next
   LLM iteration
  3. Best Code Selection - Track globally best performing reward
  function


## Critical Function: Dynamic Code Injection (eureka.py:147-178)

  ### Parse function signature from LLM output
  gpt_reward_signature, input_lst =
  get_function_signature(code_string)

  ### Create reward injection code
  reward_signature = [
      f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
      f"self.extras['gpt_reward'] = self.rew_buf.mean()",
      f"for rew_state in self.rew_dict: self.extras[rew_state] = 
  self.rew_dict[rew_state].mean()",
  ]

  ### Inject into environment's compute_reward method
  if "def compute_reward(self)" in task_code_string:
      task_code_string_iter = task_code_string.replace(
          "def compute_reward(self):",
          "def compute_reward(self):\n" + reward_signature
      )


## Data Flow:

  1. Environment Context → LLM Prompt → Generated Code
  2. Generated Code → Environment Injection → RL Training
  3. Training Metrics → TensorBoard Logs → LLM Feedback


## Design Patterns

Template Method Pattern:

  - Base environment classes provide structure (def 
  compute_reward(self))
  - Generated functions fill in the algorithmic details
  - Consistent interface maintained across all tasks

##  Q1: How does Eureka generate and inject reward functions?

Generation Process:
  1. Context Assembly - Task description + environment observation
  code → LLM prompt
  2. Function Generation - LLM produces @torch.jit.script decorated
  reward functions
  3. Code Parsing - Regex extraction of Python code from LLM response
  4. Signature Analysis - AST parsing to extract function name and
  parameters
  5. Dynamic Injection - String replacement to insert generated code
  into environment

## Q: Iterative Feedback Loop Implementation
### LLM Input Structure:
  messages = [
      {"role": "system", "content": initial_system},    # Task context
   + constraints
      {"role": "user", "content": initial_user},        # Environment 
  description
      {"role": "assistant", "content": generated_code}, # Previous 
  iteration's output
      {"role": "user", "content": feedback}             # Training 
  results + improvement suggestions
  ]


### Feedback Content (eureka.py:240-277):
  - Performance Metrics: Success rates, episode lengths, reward
  correlations
  - Component Analysis: Individual reward term statistics (mean, max,
  min)
  - Improvement Suggestions: Based on pattern analysis of training
  curves
  - Error Messages: Execution errors with full traceback when code
  fails


### Iterative Refinement:
  1. Success Rate Tracking - If near zero, complete rewrite suggested
  2. Component Magnitude Analysis - Rescaling suggestions for
  imbalanced terms
  3. Gradient Signal Analysis - Components with flat curves get
  modification suggestions
  4. Correlation Metrics - Alignment between generated and
  ground-truth rewards


### reward_signature.txt
- The reward_signature.txt
  file is located at:

`/Users/winnieyangwn/Docum
  ents/Winnie/Eureka/eureka
  /utils/prompts/reward_sig
  nature.txt`

  I showed you the contents
   earlier - it contains
  the template function
  signature:

  @torch.jit.script
  def compute_reward(object
  _pos: torch.Tensor, 
  goal_pos: torch.Tensor) 
  -> Tuple[torch.Tensor, 
  Dict[str, torch.Tensor]]:
      ...
      return reward, {}

  This file is loaded in
  eureka.py:53 and used as
  a template to show the
  LLM the expected
  structure for generated
  reward functions. It gets
   inserted into the system
   prompt via the format
  string substitution on
  line 57:

  initial_system =
  initial_system.format(tas
  k_reward_signature_string
  =reward_signature) +
  code_output_tip
