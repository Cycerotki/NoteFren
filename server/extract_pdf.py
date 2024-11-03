import pdfplumber
import pandas as pd
 
pdf = pdfplumber.open("pdf/2e6caff67e5550c07775014eadfd481b39fc.pdf")

df = pd.DataFrame(pdf.pages[1].extract_table())