import base64
import pandas as pd
from io import BytesIO

def parse_contents(contents):
    """
    Parse uploaded CSV contents into a DataFrame.
    Returns (df, error_message)
    """
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(BytesIO(decoded).read().decode('utf-8'))
        return df, None
    except Exception as e:
        return None, str(e)
