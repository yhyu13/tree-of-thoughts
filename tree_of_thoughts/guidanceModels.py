import guidance
from .abstractLanguageModel import AbstractLanguageModel
import time
import os
import openai

class GuidanceLanguageModel(AbstractLanguageModel):
    def __init__(self, model, strategy="cot", evaluation_strategy="value", enable_ReAct_prompting=False, 
                 system_prompt="You are an assitant follow exact order from user prompt"):
        # gpt4 = guidance.llms.OpenAI("gpt-4")
        # vicuna = guidance.llms.transformers.Vicuna("your_path/vicuna_13B", device_map="auto")
        self.model = model
        
        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.ReAct_prompt = '''{{#assistant~}}
            {{gen 'Observation' temperature=0.5 max_tokens=50}}
            {{~/assistant}}'''
            
        self.system_prompt = system_prompt
        
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        
        self.thoughts_program = guidance('''
            {{#system~}}
            %s
            {{~/system}}

            {{#user~}}
            Given the current state of reasoning:
            {{state_text}},
            and initial prompt:
            {{initial_prompt}},
            Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain.
            Generate {{k}} coherent thoughts as short as possible to continue the reasoning process.
            Output just thought, no courtesy, and NOTHING ELSE.
            {{~/user}}

            %s
            
            {{#assistant~}}
            {{gen 'Thoughts' temperature=0.5 max_tokens=50}}
            {{~/assistant}}
            ''' % (self.system_prompt, self.ReAct_prompt), llm=self.model)
        
        self.value_program = guidance('''
            {{#system~}}
            %s
            {{~/system}}

            {{#user~}}
            Given the current state of reasoning:
            {{state_text}},
            and initial prompt:
            {{initial_prompt}},
            Evaluate its value as a float between 0 and 1, where 0 means impossible and 1 means every likely.
            Output just float number and NOTHING ELSE.
            {{~/user}}

            {{#assistant~}}
            {{gen 'Value' temperature=0 max_tokens=10}}
            {{~/assistant}}
            ''' % self.system_prompt, llm=self.model)
        
        self.vote_program = guidance('''
            {{#system~}}
            %s
            {{~/system}}

            {{#user~}}
            Given the current state of reasoning:
            {{state_text}},
            and initial prompt:
            {{initial_prompt}},
            Vote for the best state by repeating and NOTHING ELSE.
            {{~/user}}

            {{#assistant~}}
            {{gen 'Vote' temperature=0 max_tokens=50}}
            {{~/assistant}}
            ''' % self.system_prompt, llm=self.model)
        
        self.solution_program = guidance('''
            {{#system~}}
            %s
            {{~/system}}
            
            {{#user~}}
            Considering the reasoning provided:
            {{state_text}},
            and initial prompt:
            {{initial_prompt}},
            Devise the best possible solution for the task and NOTHING ELSE.
            {{~/user}}

            {{#assistant~}}
            {{gen 'Solution' temperature=0 max_tokens=50}}
            {{~/assistant}}
            ''' % self.system_prompt, llm=self.model)
        
    def model_response_handler(self, program, **kargs):
        print("Calling guidance model(Modify Me to handle specific LLM response excpetions!)")
        reponse = program(**kargs)
        return reponse

    def generate_thoughts(self, state, k, initial_prompt):
        #implement the thought generation logic using self.model
        state_text = ' '.join(state)
        
        thoughts = []
        for _ in range(k):
            response = self.model_response_handler(self.thoughts_program, state_text=state_text, k=1, initial_prompt=initial_prompt)
            text = response['Thoughts']
            thoughts += [text]
        print(f"Generated thoughts: {thoughts}")
        return thoughts

    def evaluate_states(self, states, initial_prompt):
        #implement state evaluation logic using self.model
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                response = self.model_response_handler(self.value_program, state_text=state_text, initial_prompt=initial_prompt)
                try:
                    value_text = response['Value']
                    print(f"Value text {value_text}")
                    value = float(value_text)
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
            response = self.model_response_handler(self.vote_program, states_text=states_text, initial_prompt=initial_prompt)
            best_state_text = response['Vote']
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            print(f'best_state: {best_state}')
            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
        
    def generate_solution(self, initial_prompt, state):
        if (type(state) == str):
            states_text = state
        else:
            states_text = '\n'.join(state)
        
        response = self.model_response_handler(self.solution_program, states_text=states_text, initial_prompt=initial_prompt)
        answer = response['Solution']
        print(f"General solution : {answers}")
        return answer

class GuidanceOpenAILanguageModel(GuidanceLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=False, 
                 system_prompt="You are an assitant follow exact order from user prompt"):
        if api_key == "" or api_key == None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")

        if api_base == ""or api_base == None:
            api_base = os.environ.get("OPENAI_API_BASE", "")  # if not set, use the default base path of "https://api.openai.com/v1"
        if api_base != "":
            # e.g. https://api.openai.com/v1/ or your custom url
            openai.api_base = api_base
            print(f'Using custom api_base {api_base}')
            
        if api_model == "" or api_model == None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"
        print(f'Using api_model {self.api_model}')

        super().__init__(guidance.llms.OpenAI(self.api_model), strategy, evaluation_strategy, enable_ReAct_prompting, system_prompt)
        
    
    def model_response_handler(self, program, **kargs):
        error_msg = ''
        while True:
            try:
                program.llm.max_retries = 60
                guidance.llms.OpenAI.cache.clear()
                response = program(**kargs)
                return response
            except openai.error.RateLimitError as e:
                sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                time.sleep(sleep_duratoin)
            except Exception as e:
                if str(e) == f'''Too many (more than {guidance.llm.max_retries}) OpenAI API RateLimitError's in a row!''':
                    sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                    print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                    time.sleep(sleep_duratoin)
                else:
                    error_msg = str(e)
                    break
        raise Exception(error_msg)
