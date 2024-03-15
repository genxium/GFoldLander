A temporary pandoc command to convert directly into docx.

```
pandoc --metadata-file pandoc-metadata.yaml --reference-doc ../AMS/AMS-template.docx -s manuscript.tex -o manuscript_pandoc.docx --citeproc
```

or 

```
pandoc --metadata-file pandoc-metadata.yaml -F pandoc-crossref --reference-doc ../AMS/AMS-template.docx -s manuscript.tex -o manuscript_pandoc.docx --citeproc
```

with the `crossref` filter. 

The `AMS-template.docx` is used to automatically fulfill the following requirements
- header & footer sections 
- page size & margins
- fonts 
