from PyPDF2 import PdfReader

reader = PdfReader("input/hitopadesh.pdf")
number_of_pages = len(reader.pages)
print(number_of_pages)
page = reader.pages[28]
text = page.extract_text()

print(text)