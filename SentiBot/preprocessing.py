import pandas as pd
from langchain_core.documents import Document


def dataconveter():
    product_data=pd.read_csv(r"/Users/rasikagulhane/Desktop/Customer_Rating_Reviews/data/modeified_raw_reviews.csv")

    data=product_data

    product_list = [['Product_ID','Age','Title','Text_Review','Rating','Type','sentiment','sentiment_score','rating_match']]




    # Iterate over the rows of the DataFrame
    for index, row in data.iterrows():
        # Construct an object with 'product_name' and 'review' attributes
        obj = {


                'product_id': row['Product_ID'],
                'age': row['Age'],
                'review_title': row['Title'],
                'review':row['Text_Review'],
                'ratings':row['Rating'],
                'product_type': row['Type'],
                'sentiment': row['sentiment'],
                'sentiment_score': row['sentiment_score'],
                'rating_match': row['rating_match']
            
             
            }
        # Append the object to the list
        product_list.append(obj)

        
            
    docs = []
    for entry in product_list[1:]:  # Skip the first item which contains the column headers
        metadata = {
            'product_id': entry['product_id'],
            'age': entry['age'],
            'review_title': entry['review_title'],
            'ratings': entry['ratings'],
            'product_type': entry['product_type'],
            'sentiment': entry['sentiment'],
            'sentiment_score': entry['sentiment_score'],
            'rating_match': entry['rating_match']
    }
        doc = Document(page_content=entry['review'], metadata=metadata)
        docs.append(doc)

    return docs