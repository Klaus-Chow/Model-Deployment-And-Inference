#coding:UTF-8
path =  '/src/notebooks/crnnmobile/key.txt'
alphabet =''.join([ x.strip('\n') for x in list(open(path,'r',encoding='utf-8'))])
alphabet = alphabet.replace('\ufeff','').replace('\u3000','').strip()
 