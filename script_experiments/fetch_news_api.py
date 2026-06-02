import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from newsapi import NewsApiClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the current working directory of the script
script_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing

def get_data(query, start_date, end_date, sort_by, page_size, language='en'):
    """
    Fetch news articles from News API based on search parameters.
    
    Args:
        query (str): Keywords or phrases to search for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        sort_by (str): Sort order - 'relevancy', 'popularity', or 'publishedAt'
        page_size (int): Number of results to return (max 100)
        language (str): Language code (default: 'en')
    
    Returns:
        pd.DataFrame: DataFrame containing article data
    """
    print("\n" + "="*40)
    print("FETCHING NEWS ARTICLES")
    print("="*40)
    
    # Get API key from environment
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise ValueError("NEWS_API_KEY not found in environment variables")
    
    # Initialize News API client
    newsapi = NewsApiClient(api_key=api_key)
    
    print(f"Query: {query}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Sort By: {sort_by}")
    print(f"Page Size: {page_size}")
    print(f"Language: {language}")
    
    # Fetch articles
    all_articles = newsapi.get_everything(
        q=query,
        language=language,
        from_param=start_date,
        to=end_date,
        sort_by=sort_by,
        page_size=page_size
    )
    
    print(f"\nTotal Results: {all_articles.get('totalResults', 0)}")
    print(f"Articles Retrieved: {len(all_articles.get('articles', []))}")
    
    # Convert to DataFrame
    articles_df = pd.DataFrame(all_articles['articles'])
    
    print(f"\nShape: {articles_df.shape}")
    print(f"\nColumns: {list(articles_df.columns)}")
    print(f"\nPreview:\n{articles_df.head(3)}\n")
    
    return articles_df

def save_data(articles_df, save_path, prefix=None):
    """
    Save articles DataFrame to file.
    
    Args:
        articles_df (pd.DataFrame): DataFrame containing article data
        save_path (str): Directory path to save the file
        prefix (str): Optional prefix for the filename
    """
    print("\n" + "="*40)
    print("SAVING DATA")
    print("="*40)
    
    # Create filename with timestamp if no prefix provided
    if prefix is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        prefix = f"news_articles_{timestamp}.csv"
    elif not prefix.endswith('.csv'):
        prefix = f"{prefix}.csv"
    
    print(f"Save Path: {save_path}")
    print(f"Filename: {prefix}")
    print(f"Rows to Save: {len(articles_df)}")
    
    # Save using DataProcessing utility
    DataProcessing.save_to_file(
        data=articles_df,
        path=save_path,
        prefix=prefix,
        save_file_type='csv'
    )
    
    print(f"✓ Data saved successfully")

if __name__ == "__main__":
    """
    Usage examples:
    
    # E1: Fetch Tesla and autonomous vehicle articles
    python fetch_news_api.py \
        --query "(team OR season OR playoff) AND (predict OR expected OR likely)" \
        --start_date 2026-03-21 \
        --end_date 2026-03-28 \
        --sort_by relevancy \
        --page_size 100
    
    # E2: Fetch Bitcoin articles sorted by popularity
    python fetch_news_api.py \
        --query "bitcoin OR cryptocurrency" \
        --start_date 2026-02-01 \
        --end_date 2026-02-28 \
        --sort_by popularity \
        --page_size 50
    
    # E3: Fetch prediction-related articles sorted by date
    python fetch_news_api.py \
        --query "forecast OR predict OR expected OR outlook" \
        --start_date 2026-01-01 \
        --end_date 2026-01-31 \
        --sort_by publishedAt \
        --page_size 75
    """
    

    
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_save_path = os.path.join(base_data_path, 'news_api')
    
    # Create save directory if it doesn't exist
    os.makedirs(default_save_path, exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description='Fetch news articles from News API and save to file'
    )
    
    parser.add_argument(
        '--query',
        required=True,
        help='Search keywords or phrases. Examples: "Tesla", "bitcoin OR cryptocurrency", '
             '"forecast OR predict OR expected", "climate change AND policy"'
    )
    
    parser.add_argument(
        '--start_date',
        required=True,
        help='Start date in YYYY-MM-DD format. Example: 2026-03-01'
    )
    
    parser.add_argument(
        '--end_date',
        required=True,
        help='End date in YYYY-MM-DD format. Example: 2026-03-31'
    )
    
    parser.add_argument(
        '--sort_by',
        default='relevancy',
        choices=['relevancy', 'popularity', 'publishedAt'],
        help='Sort order for articles. Options: "relevancy" (default), "popularity", "publishedAt"'
    )
    
    parser.add_argument(
        '--page_size',
        type=int,
        default=100,
        help='Number of results to return (max 100). Default: 100'
    )
    
    parser.add_argument(
        '--save_path',
        default=default_save_path,
        help=f'Directory to save the output file. Default: {default_save_path}'
    )
    
    parser.add_argument(
        '--output_filename',
        default=None,
        help='Custom output filename (without extension). If not provided, uses timestamp.'
    )
    
    args = parser.parse_args()
    
    # Validate page_size
    if args.page_size > 100:
        print("Warning: page_size exceeds maximum of 100. Setting to 100.")
        args.page_size = 100
    
    # ============================================================
    # 2. FETCH DATA
    # ============================================================
    articles_df = get_data(
        query=args.query,
        start_date=args.start_date,
        end_date=args.end_date,
        sort_by=args.sort_by,
        page_size=args.page_size
    )
    
    # ============================================================
    # 3. SAVE DATA
    # ============================================================
    save_data(
        articles_df=articles_df,
        save_path=args.save_path,
        prefix=args.output_filename
    )
    
    # ============================================================
    # 4. COMPLETE
    # ============================================================
    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print("="*40)
    print(f"Total articles saved: {len(articles_df)}")
    print(f"Location: {args.save_path}")