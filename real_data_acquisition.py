# """Detravious Jamari Brinkley, Kingdom Man

# Dr. Grant @ UF Data Studio


# """
# import requests

# from abc import abstractmethod

# from data_processing import DataProcessing
# from text_generation_models import TextGenerationModelFactory

# class OpenMeasuresBuilder:
#     """A class to execute the building process. Each function is like a section (0341s, 0311s) that have a specific focus in the process.
    
#     Process follows notebook: https://colab.research.google.com/drive/1kDyRIC0NBOj4Egn_VdK837QBNqDERRi_?usp=sharing
#     """

#     @abstractmethod
#     def reset(self):
#         pass

#     @abstractmethod
#     def generate_terms(self, prompt):
#         """Use X LLM to create terms"""
#         tgmf = TextGenerationModelFactory()

#         # Groq Cloud (https://console.groq.com/docs/overview)
#         gemma_29b_generation_model = tgmf.create_instance('gemma2-9b-it') 
#         input_prompt = gemma_29b_generation_model.user(prompt)
#                 # print(input_prompt)  
                
#         raw_text_llm_generation = gemma_29b_generation_model.chat_completion([input_prompt])
#         # print(raw_text_llm_generation)
#         # print("====================================")
#         for line in raw_text_llm_generation.split("\n"):
#             # print(line)
#             if line.strip():
#                 print(f"search query: {line}")
#         self.terms_for_query = line

#     @abstractmethod
#     def set_query(self, limit: int, site: str, start_date: str, end_date: str, querytype: str):
#         """Required parameters to query Open Measure"""

#         self.params = {
#             'term' : self.terms_for_query,
#             'limit': limit,
#             'site': site,
#             'since': start_date,
#             'until': end_date,
#             'querytype': querytype
#             }
        
#         # we can create a URL to represent this query
#         self.url = 'http://api.smat-app.com/content?{}'.format(
#             '&'.join(
#                 [f"{k}={v}" for k,v in self.params.items()]
#             )
#         )

#         print(f"Query's URL: {self.url}")
                
#     @abstractmethod
#     def get_raw_hits(self):
#         r = requests.get(self.url)
#         r.status_code
#         data = r.json()
#         self.hits = data['hits']['hits']

#         return  self.hits
    
#     def convert_raw_hits_to_df(self):
#         df = DataProcessing.convert_to_df(data=self.hits, mapping='Open Measures')
#         return df

# class OpenMeasuresDirector:
#     """A class to orchestrate the building process. Think at large/overview/blue print or this is like the CO laying the plan to the entire company. The execution (Builder) has functions and these functions are different sections (0341s, 0311s, etc) acting."""

#     def construct_from_dataset(prompt, limit: int, site: str, start_date: str, end_date: str, querytype: bool, regenerations: int):
#         """Construct all datasets"""

#         builder = OpenMeasuresBuilder()
#         builder.reset()
#         builder.generate_terms(prompt)
#         builder.set_query(limit, site, start_date, end_date, querytype)
#         builder.get_raw_hits()
#         df = builder.convert_raw_hits_to_df()
#         df['Site'] = site
#         return df


import pdb
import requests
from abc import ABC, abstractmethod
from data_processing import DataProcessing
from text_generation_models import TextGenerationModelFactory

class OpenMeasuresBuilder(ABC):
    def __init__(self):
        self.terms_for_query = []
        self.params = {}
        self.url = None
        self.hits = []
        self.feedback = None

    @abstractmethod
    def reset(self):
        pass

    def generate_terms(self, prompt, model_name='gemma2-9b-it'):
        """
        Generate terms with boolean logic using an LLM based on the provided prompt and model name.

        Parameters
        ----------
        prompt : str
            The input prompt to generate terms from.
        model_name : str, optional
            The name of the LLM model to use (default is 'gemma2-9b-it').

        Returns
        -------
        None

        Notes
        -----
        This method generates terms using the specified LLM model based on the input prompt.
        It stores the generated terms in `self.terms_for_query` and updates the `self.params['term']`
        parameter accordingly. If there is only one term, it is converted to a string; otherwise,
        the list of terms is used directly.
        """
        self.prompt = prompt
        self.model_name = model_name
        self.terms_by_prompt = {}  # Store terms per prompt

        tgmf = TextGenerationModelFactory()
        generation_model = tgmf.create_instance(model_name)
        input_prompt = generation_model.user(self.prompt)
        raw_text_llm_generation = generation_model.chat_completion([input_prompt])

        # Initialize the list for this prompt
        self.terms_by_prompt[self.prompt] = []

        for line in raw_text_llm_generation.split("\n"):
            if line.strip():
                print(f"\tTerm for query: {line.strip()}\n")
                self.terms_for_query.append(line.strip())

        # Update self.params['term'] based on the number of generated terms as self.params['term'] only takes string
        if len(self.terms_for_query) == 1: 
            # single, so turn into string so we can pass directly into self.params['term'] as it takes a string
            self.params['term'] = ' '.join(self.terms_for_query)
        else:
            # multiple, so keep and write logic to turn each into string so we can pass directly into self.params['term'] as it takes a string and not a list
            # self.params['term'] = self.terms_for_query 
            
            # multiple, so turn into string so we can pass directly into self.params['term'] as it takes a string --- get last string
            self.params['term'] = self.terms_for_query[-1]
            print(f"\tUpdated terms: {self.params['term']}\n")
        # print(f"\tAll terms for query: {self.terms_for_query}\n")

    def set_query(self, limit: int, site: str, start_date: str, end_date: str, querytype: str):
        """Set query parameters including terms, prompt, and model"""
        self.params.update({
            'limit': limit,
            'site': site,
            'since': start_date,
            'until': end_date,
            'querytype': querytype
        })

        self.url = 'http://api.smat-app.com/content?{}'.format(
            '&amp;'.join([f"{k}={v}" for k, v in self.params.items()])
        )
        print(f"\tQuery's URL: {self.url}\n")


    def get_raw_hits(self):
        r = requests.get(self.url)
        if r.status_code == 200:
            data = r.json()
            self.hits = data['hits']['hits']
            if not self.hits:
                print(f"\tNo hits (404): {self.hits}")
            else:
                # print(f"    Hits: {self.hits[0]['text']}")
                return self.hits
        else:
            print(f"\tNo hits (404): Failed to retrieve data: {r.status_code}")
            return []

    def convert_raw_hits_to_df(self):
        # print(f"hits: {type(self.hits)}")
        self.params['prompt'] = self.prompt
        self.params['model'] = self.model_name
        df = DataProcessing.convert_to_df(data=self.hits, mapping='Open Measures')
        df['Query Params'] = [self.params] * len(df)
        return df

    def refine_query_based_on_hits(self, feedback):
        if feedback.lower() != "accept":
            # Append the feedback to the original prompt
            self.prompt += f" {feedback}"
            print(f"\t\t--->Updated prompt: {self.prompt}\n\tNew terms with updated prompt")
            # Generate terms using the updated prompt
            self.generate_terms(self.prompt)

class ConcreteOpenMeasuresBuilder(OpenMeasuresBuilder):
    def reset(self):
        self.__init__()

class OpenMeasuresDirector:
    @staticmethod
    def construct_from_dataset(prompt, limit: int, site: str, start_date: str, end_date: str, querytype: str, model_name='gemma2-9b-it'):
        print(f"=============================== Site: {site} ===============================")

        builder = ConcreteOpenMeasuresBuilder()

        print("### RESET ###")
        builder.reset()

        print("### GENERATE TERMS ###")
        builder.generate_terms(prompt, model_name)

        print("### SET QUERY ###")
        builder.set_query(limit, site, start_date, end_date, querytype)

        print("### REGENERATE QUERY ###")
        hits_per_site_dfs = OpenMeasuresDirector.refine_query_loop(builder, site)
    
        return hits_per_site_dfs

    @staticmethod
    def refine_query_loop(builder, site):
        hits_per_site_dfs = []
        regenerations_idx = 0
        while True:
            print(f"\n------\n\n\tRegenerations: {regenerations_idx}")
            hits = builder.get_raw_hits()
            
            if hits:
                df = builder.convert_raw_hits_to_df()
                df['Site'] = site
                hits_per_site_dfs.append(df)
                
                print("Hits retrieved:")
                print(df)  # Display the hits to the user
            
            feedback = input("Do you want to refine the query? If yes, please provide your feedback: ")
            builder.refine_query_based_on_hits(feedback.lower())
            
            if feedback.lower() == "accept":
                break
            
            regenerations_idx += 1
        
        return hits_per_site_dfs

        # print()

        # builder.get_raw_hits()

        # hits_per_site_dfs = []

        # for regenerations_idx in range(regenerations):
        #     print(f"   Regenerations: {regenerations_idx}")
        #     hits = builder.get_raw_hits()
            
        #     if hits:
        #         df = builder.convert_raw_hits_to_df()
        #         df['Site'] = site
        #         hits_per_site_dfs.append(df)
                
        #         print("Hits retrieved:")
        #         print(df)  # Display the hits to the user
                
        #     feedback = input("Do you want to refine the query? If no, type 'accept' and if yes, please provide your feedback: ")
        #     builder.refine_query_based_on_hits(feedback.lower())
            
        #     if feedback.lower() == "accept":
        #         break

        # print("==============================================================")

        # return hits_per_site_dfs