try:
    from PIL import Image
except ImportError:
    import Image
import cv2
import pytesseract
import os
import numpy as np
import pandas as pd
import re
from pdf2image import convert_from_bytes

# Some help functions 
def get_conf(page_gray):
    '''return a average confidence value of OCR result '''
    df = pytesseract.image_to_data(page_gray,output_type='data.frame')
    df.drop(df[df.conf==-1].index.values,inplace=True)
    df.reset_index()
    return df.conf.mean()
  
def deskew(image):
    '''deskew the image'''
    gray = cv2.bitwise_not(image)
    temp_arr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(temp_arr > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def combine_texts(list_of_text):
    '''Taking a list of texts and combining them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text
def extract_high_conf_words(list_of_confs,list_of_words):
    combined_text =''
    for i in range(len(list_of_confs)):
        if list_of_confs[i] >  90:
            combined_text += ' '+list_of_words[i]
            # print('conf=',list_of_confs[i],',word =',list_of_words[i])
    return combined_text
'''
Main part of OCR:
pages_df: save extracted text for each pdf file, index by page
OCR_dic : dict for saving df of each pdf, filename is the key
'''

# %%time 

OCR_dic={} 
PATH = 'input'
file_list = ['hitopadesh-25-29.pdf']
def extract_file():
    for file in file_list:
        # convert pdf into image
        pdf_file = convert_from_bytes(open(os.path.join(PATH,file), 'rb').read())
        # create a df to save each pdf's text
        pages_df = pd.DataFrame(columns=['conf','text'])
        for (i,page) in enumerate(pdf_file) :
            try:
                # transfer image of pdf_file into array
                page_arr = np.asarray(page)
                # transfer into grayscale
                page_arr_gray = cv2.cvtColor(page_arr,cv2.COLOR_BGR2GRAY)
                page_arr_gray = cv2.fastNlMeansDenoising(page_arr_gray,None,3,7,21)
                page_deskew = deskew(page_arr_gray)
                # cal confidence value
                page_conf = get_conf(page_deskew)
                # extract string 
                d = pytesseract.image_to_data(page_deskew,output_type=pytesseract.Output.DICT)
                # print('====================TEXT ================',len(d['text']),'============ level=',len(d['level']),'=== conf=',len(d['conf']),'======')
                # print(d['text'])
                pagetext = extract_high_conf_words(d['conf'],d['text'])
                print(pagetext)
                d_df = pd.DataFrame.from_dict(d)
                # get block number
                block_num = int(d_df.loc[d_df['level']==2,['block_num']].max())
                # drop header and footer by index
                head_index = d_df[d_df['block_num']==1].index.values
                foot_index = d_df[d_df['block_num']==block_num].index.values
                # d_df.drop(head_index,inplace=True)
                # d_df.drop(foot_index,inplace=True)
                # combine text in dataframe
                # text = combine_texts(d_df.loc[d_df['level']==5,'text'].values)
                pages_df = pages_df.append({'conf': page_conf,'text': text}, ignore_index=True)
            except Exception as e:
                # if can't extract then give some notes into df
                if hasattr(e,'message'):
                    pages_df = pages_df.append({'conf': -1,'text': e.message}, ignore_index=True)
                else:
                    pages_df = pages_df.append({'conf': -1,'text': e}, ignore_index=True)
                continue
    # save df into a dict with filename as key        
    OCR_dic[file]=pages_df
    # print(text)
    print('{} is done'.format(file))


        
extract_file()