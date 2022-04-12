import jsonlines

def store_text_to_file(f_path, text):
  with open(f_path, 'w') as f:
    text = f.write(text)

def jsonline_to_html_file(f_path):
    with jsonlines.open(f_path) as reader:
        for obj in reader:
            f_name = obj['url'].split('//')[-1]
            f_name = f_name.replace('/', '-')
            f_name = 'tmp/' + f_name + ".html"

            store_text_to_file(f_name, obj['text'])

script = """
<script>console.log('hello!')</script>
<style>
    * {
        border: 1px solid #eee;
        padding: 10px;
        box-sizing: content-box;
        background-color: #fff;
        color: #000;
    }
    main:hover > *, section:hover > *, header:hover > *, nav:hover > *, a:hover > *, div:hover > *, p:hover > *, ul:hover > *, h1:hover > *, h2:hover > *, h3:hover > *, li:hover > *, span:hover > *{
        border: 2px solid green;
        background-color: #fffeee;
        padding: 10px 29px;
        margin: 5px 0;
        color: green;
    }
    main:hover, section:hover, header:hover, nav:hover, a:hover, div:hover, p:hover, ul:hover, h1:hover, h2:hover, h3:hover, li:hover, span:hover{
        border: 1px solid #fff;
        background-color: #eee;
        padding: 17px;
    }
    a, span {
        display: block;
    }
</style>
"""

jsonline_to_html_file('raw_html_spider.jl')
