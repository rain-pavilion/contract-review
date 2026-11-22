import importlib, sys
print("EXEC", sys.executable, sys.version.split()[0])
mods = ['numpy','pandas','docx','PIL','lxml','requests','pymupdf','fitz','sklearn','torch',
        'jieba','paddleocr','rapidocr_onnxruntime','pytesseract','cv2','onnxruntime',
        'transformers','langgraph','networkx','scipy','openpyxl','pptx','bs4','tqdm','streamlit','qdrant_client']
for m in mods:
    try:
        mod = importlib.import_module(m)
        print('%-26s OK   %s' % (m, getattr(mod, '__version__', '')))
    except Exception as e:
        print('%-26s --   %s' % (m, type(e).__name__))
