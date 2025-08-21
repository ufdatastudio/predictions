"""Detravious Jamari Brinkley, Kingdom Man

Dr. Grant @ UF Data Studio


"""
import requests

from abc import abstractmethod

from data_processing import DataProcessing

class OpenMeasuresBuilder:
    """A class to execute the building process. Each function is like a section (0341s, 0311s) that have a specific focus in the process.
    
    Process follows notebook: https://colab.research.google.com/drive/1kDyRIC0NBOj4Egn_VdK837QBNqDERRi_?usp=sharing
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_terms(self, terms):
        # type is dict
        if isinstance(terms, dict):
            pass
        
        elif isinstance(terms, str):
            self.terms_for_query = terms
            # self.terms_for_query = 

        # type is list



    @abstractmethod
    def set_query(self, limit: int, site: str, start_date: str, end_date: str, querytype: str):
        """Required parameters to query Open Measure"""

        self.params = {
            'term' : self.terms_for_query,
            'limit': limit,
            'site': site,
            'since': start_date,
            'until': end_date,
            'querytype': querytype
            }
        
        # we can create a URL to represent this query
        self.url = 'http://api.smat-app.com/content?{}'.format(
            '&'.join(
                [f"{k}={v}" for k,v in self.params.items()]
            )
        )

        print(f"Query's URL: {self.url}")
                
    @abstractmethod
    def get_raw_hits(self):
        r = requests.get(self.url)
        r.status_code
        data = r.json()
        self.hits = data['hits']['hits']

        return  self.hits
    
    def convert_raw_hits_to_df(self):
        df = DataProcessing.convert_to_df(data=self.hits, mapping='Open Measures')
        return df

class OpenMeasuresDirector:
    """A class to orchestrate the building process. Think at large/overview/blue print or this is like the CO laying the plan to the entire company. The execution (Builder) has functions and these functions are different sections (0341s, 0311s, etc) acting."""

    def construct_from_dataset(terms, limit: int, site: str, start_date: str, end_date: str, querytype: bool):
        """Construct all datasets"""
        
        builder = OpenMeasuresBuilder()
        builder.reset()
        builder.set_terms(terms)
        builder.set_query(limit, site, start_date, end_date, querytype)
        builder.get_raw_hits()
        df = builder.convert_raw_hits_to_df()
        df['Site'] = site
        return df


