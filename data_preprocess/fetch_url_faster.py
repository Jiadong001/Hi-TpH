import requests
from bs4 import BeautifulSoup
import re
import json
import os

import asyncio
import aiohttp

import pandas as pd

# prepare urls
save_path = "/data/lujd/TCRdata/collect//"

## IEDB vseq
savename = "IEDB.csv"
df = pd.read_csv(save_path+savename, sep=',', low_memory=False)
i = 17              # fetch in batches, change i manually: 0-17
batch = 10000
iedb_urls = list(df["iedb.url"].unique())[batch*i: batch*(i+1)]    # total: 176238 (receptor_table_export_1694687913.csv: downloaded from IEDB on 2023.9.14)

# Function to async fetch the content of a single URL
async def fetch_url_txt_async(url):
    try:
        response = await asyncio.to_thread(requests.get, url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            return response.text        # str, or you can also use content
        else:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching {url} asynchronously: {e}")

async def fetch_url_content_async(url):
    try:
        response = await asyncio.to_thread(requests.get, url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            return response.content     # byte -> for writting in
        else:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching {url} asynchronously: {e}")

# Function to extract JavaScript variables from a script block
def extract_js_variables(script_content):
    # Define a regular expression pattern to find JavaScript variables
    pattern = r'var\s+(\w+)\s*=\s*(.*?);'
    matches = re.findall(pattern, script_content)
    
    # Create a dictionary to store variable names and their values
    variables = {}
    for match in matches:
        variable_name, variable_value = match
        variables[variable_name] = variable_value
    
    return variables

# Main function to get V domain sequence from given iedb url
async def get_vseq():
    tasks = [fetch_url_content_async(url) for url in iedb_urls]
    results = await asyncio.gather(*tasks)

    # Process the results as needed
    for url, content in zip(iedb_urls, results):
        alpha_vseq = None
        beta_vseq = None

        if content is not None:
            soup = BeautifulSoup(content, "html.parser")
            
            # Find all script tags with type="text/javascript"
            script_tags = soup.find_all("script", type="text/javascript")
            
            for script_tag in script_tags:
                script_content = script_tag.string
                if script_content:
                    # Extract JavaScript variables from the script content
                    variables = extract_js_variables(script_content)
                    
                    # Get the target variable
                    if 'receptorTableData' in variables.keys():
                        # Use json.loads() to parse the string into a dictionary
                        TableData = json.loads(variables['receptorTableData'])

                        # get chain1vdomseq & chain2vdomseq
                        alpha_vseq = TableData['data']['receptorsData'][0]['chain1vdomseq']     # Sometimes there are 2 vseqs, [0] indicates we keep the first
                        beta_vseq = TableData['data']['receptorsData'][0]['chain2vdomseq']

        print(f"{url},{alpha_vseq},{beta_vseq}")


if __name__ == "__main__":
    # get v domain sequences
    # save format: [url, alpha.vseq, beta.vseq]
    asyncio.run(get_vseq())
    # print("---")
