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

import pandas as pd 

from abc import ABC, abstractmethod

from data_processing import DataProcessing
from text_generation_models import TextGenerationModelFactory

class OpenMeasuresBuilder(ABC):
    """
    OpenMeasures (https://openmeasures.io/open-source/) is an open-source platform that stores (social media) data and allows API access to work with data.
    Using the Design Pattern | Creational Pattern | Builder (https://refactoring.guru/design-patterns/builder) = a step by step (assembly line like) to construct objects. 
    A class to execute the building process. Each function is a step in the process and has a specific focus.
    Process follows notebook: https://colab.research.google.com/drive/1kDyRIC0NBOj4Egn_VdK837QBNqDERRi_?usp=sharing
    """
    def __init__(self):
        self.terms_for_query = []
        self.params = {}
        self.url = None
        self.hits = []
        self.feedback = None

    @abstractmethod
    def reset(self):
        pass

    def user_specify_query_string(self, query_string) -> None:
        """
        User specify terms boolean logic.

        Parameters
        ----------
        query_string : str
            The input prompt to generate terms from.

        Returns
        -------
        None as we set/assign the initialized variables with self.
        """
        self.prompt = query_string
        self.model_name = 'user'
        self.terms_for_query.append(query_string)
        self.params['term'] = ' '.join(self.terms_for_query)

        print(f"\tQuery String: {type(self.params['term']), self.params['term']}\n")

    def llm_generate_query_string(self, query_string, model_name='gemma2-9b-it') -> None:
        """
        Generate terms with boolean logic using an LLM based on the provided prompt and model name.

        Parameters
        ----------
        query_string : str
            The input prompt to generate terms from.
        model_name : str, optional
            The name of the LLM model to use (default is 'gemma2-9b-it').

        Returns
        -------
        None as we set/assign the initialized variables with self.

        Notes
        -----
        This method generates terms using the specified LLM model based on the input prompt.
        It stores the generated terms in `self.terms_for_query` and updates the `self.params['term']`
        parameter accordingly. If there is only one term, it is converted to a string; otherwise,
        the list of terms is used directly.
        """
        self.prompt = query_string
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
                print(f"\tQuery String: {line.strip()}\n")
                self.terms_for_query.append(line.strip())

        # Update self.params['term'] based on the number of generated terms as self.params['term']The only takes string
        if len(self.terms_for_query) == 1: 
            # single, so turn into string so we can pass directly into self.params['term'] as it takes a string
            self.params['term'] = ' '.join(self.terms_for_query)
        else:
            # multiple, so keep and write logic to turn each into string so we can pass directly into self.params['term'] as it takes a string and not a list
            # self.params['term'] = self.terms_for_query 
            
            # multiple, so turn into string so we can pass directly into self.params['term'] as it takes a string --- get last string
            self.params['term'] = self.terms_for_query[-1]
            print(f"\tUpdated Query String: {self.params['term']}\n")
        # print(f"\tAll terms for query: {self.terms_for_query}\n")

    def set_query(self, limit: int, site: str, start_date: str, end_date: str, querytype: str = 'query_string') -> None:
        """
        Set query parameters including terms, prompt, and model
        
        Parameters
        ----------
        limit : int
            The max # of data to return.
        site : str
            Source of the data ('tiktok', 'bluesky', 'truth social', 'llm generated', etc).
        start_date : str
            The earliest date to collect data from.
        end_date : str
            The latest date to collect data from.
        querytype : str, optional
            The method to apply boolean logic.  
            Can use default 'query_string'. Or use content or boolean_content. 
            For default, Elasticsearch across all fields. See meaning of other two in link in Notes.

        Returns
        -------
        None as we set/assign the initialized variables with self.

        Notes
        -----
        https://docs.openmeasures.io/docs/guides/public-api
        """
        self.params.update({
            'limit': limit,
            'site': site,
            'since': start_date,
            'until': end_date,
            'querytype': querytype
        })
        self.url = 'http://api.smat-app.com/content?{}'.format(
            '&'.join(
                [f"{k}={v}" for k,v in self.params.items()]
            )
        )
        print(f"\tQuery's URL: {self.url}\n")

    def get_raw_hits(self) -> dict:
        """
        Use the url from set_query() to get hits = data related to query via searching method (querytype)
        
        Returns
        -------
        dict, None, or []
            dict: The related posts.
            None: No hits
            []: Failed to retrieve data
        """
        r = requests.get(self.url)
        if r.status_code == 200:
            data = r.json()
            self.hits = data['hits']['hits']
            if not self.hits:
                print(f"\tNo hits: {r.status_code}")
                return None
            else:
                print(f"\tHits: {r.status_code}")
                # print(f"    Hits: {self.hits[0]['text']}")
                return self.hits
        else:
            print(f"\tNo hits/Failed to retrieve data: {r.status_code}")
            return []

    def convert_raw_hits_to_df(self):
        """
        Convert from dict -> df.
        
        Returns
        -------
        df
            The hits + meta data (query params).
            This df differs per site.

        Notes
        -----
        Performing the conversion from dict -> df using DataProcessing class in data_processing.
        """
        # print(f"hits: {type(self.hits)}")
        self.params['query_string'] = self.prompt
        self.params['model'] = self.model_name
        df = DataProcessing.convert_to_df(data=self.hits, mapping='Open Measures')
        df['Query Params'] = [self.params] * len(df) # include query string, limit, ..., model to know what produced the hits
        return df

    def refine_query_based_on_hits(self, feedback):
        """
        Human-in-the-loop process as user can provide feedback to generate a better query string.
        
        Parameters
        ----------
        quefeedbackry_string : str
            The input prompt to generate a better query string.

        Returns
        -------
        None
        """
        if feedback.lower() != "accept":
            # Append the feedback to the original prompt
            self.prompt += f" {feedback}"
            print(f"\t\t--->Updated prompt: {self.prompt}\n\tNew terms with updated prompt")
            # Generate terms using the updated prompt
            self.llm_generate_query_string(self.prompt)

class ConcreteOpenMeasuresBuilder(OpenMeasuresBuilder):
    def reset(self):
        self.__init__()

class OpenMeasuresDirector:
    @staticmethod
    def construct_from_dataset(
        query_string: str,
        query_string_by: str,
        limit: int,
        site: str, 
        start_date: str,
        end_date: str,
        querytype: str = 'query_string',
        model_name='gemma2-9b-it'
        ) -> pd.DataFrame:
        """
        Construct the hits given the params
        
        Parameters
        ----------
        query_string : str, optional
            The string with boolean logic to pass as params that'll query the data of the site.
            Should either be a user defined string (skip LLM, use this string directly) or a full prompt (LLM route).
        query_string_by : str, optinal
            The method that matches the query_string param
            Should either be 'user' or 'llm'
        limit : int
            The max # of data to return.
        site : str
            Source of the data ('tiktok', 'bluesky', 'truth social', 'llm generated', etc).
        start_date : str
            The earliest date to collect data from.
        end_date : str
            The latest date to collect data from.
        querytype : str, optional
            The method to apply boolean logic.  
            Can use default 'query_string'. Or use content or boolean_content. 
            For default, Elasticsearch across all fields. See meaning of other two in link in Notes.
        model_name : str, optional
            The name of the LLM model to use (default is 'gemma2-9b-it').

        Returns
        -------
        pd.DataFrame
            The hits with meta data and more. 

        Notes
        -----
        https://docs.openmeasures.io/docs/guides/public-api
        """
        print(f"=============================== Site: {site} ===============================")

        builder = ConcreteOpenMeasuresBuilder()

        print("### RESET ###")
        builder.reset()

        if query_string_by.lower() == "user":
            print("### USER SPECIFY QUERY STRINGS ###")
            builder.user_specify_query_string(query_string)

        elif query_string_by.lower() == "llm":
            print("### LLM GENERATE QUERY STRINGS ###")
            builder.llm_generate_query_string(query_string, model_name)
            
        else:
            return f"404: query_string_by is {query_string_by} and must be either: user or llm"

        print("### SET QUERY ###")
        builder.set_query(limit, site, start_date, end_date, querytype)

        print("### REGENERATE QUERY ###")
        hits_per_site_dfs = OpenMeasuresDirector.refine_query_loop(builder, site)
    
        return hits_per_site_dfs

    @staticmethod
    def refine_query_loop(builder, site):
        """
        Human-in-loop to verify quality query prompts dynamically.
        
        Parameters
        ----------
        builder : OpenMeasuresDirector
            The class.
        site : str
            Source of the data ('tiktok', 'bluesky', 'truth social', 'llm generated', etc).

        Returns
        -------
        pd.DataFrame
            The hits with meta data and more. 

        Notes
        -----
        https://docs.openmeasures.io/docs/guides/public-api
        """
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
            
            feedback = input("Do you want to refine the query? If no, type 'accept' and if yes, please provide your feedback: ")
            builder.refine_query_based_on_hits(feedback.lower())
            
            if feedback.lower() == "accept":
                break
            
            regenerations_idx += 1
        
        return hits_per_site_dfs