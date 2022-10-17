function OpenOriginalLink($event){
    $event.stopPropagation();
    console.log(url);
}

function removeItem($event){
    $event.stopPropagation();
    console.log('Remove item: ', url)
    localStorage.removeItem(url)
}

function exportData($event){
    $event.stopPropagation();

    result = []
    for(let i=0; i<localStorage.length; i++) {
        let key = localStorage.key(i);
        result.push(localStorage.getItem(key))
    }
    const _url = URL.createObjectURL(new Blob([result.join("\n")], {type: 'text/plain'}))
    $event.target.href = _url;
    localStorage.clear()
}

var removeItemBtn = document.createElement("BUTTON");   // Create a <button> element
removeItemBtn.innerHTML = "Remove Item";
removeItemBtn.classList.add('btn');
removeItemBtn.style.position = "fixed";
removeItemBtn.style.left = '5px';
removeItemBtn.style.top = 0;
removeItemBtn.style.border = "1px solid";
removeItemBtn.style.boxShadow = "3px 3px #888";
removeItemBtn.onclick = removeItem;
document.body.appendChild(removeItemBtn);

var goToOriginLink = document.createElement("A");   // Create a <button> element
goToOriginLink.innerHTML = "GoToOrigin";
goToOriginLink.classList.add('btn');
goToOriginLink.style.position = "fixed"
goToOriginLink.style.left = '115px'
goToOriginLink.style.top = 0;
goToOriginLink.onclick = OpenOriginalLink;
goToOriginLink.setAttribute('href', decodeURIComponent(url));
goToOriginLink.target = '_blank';
document.body.appendChild(goToOriginLink);

var current = new Date
var exportDataLink = document.createElement("A");   // Create a <button> element
exportDataLink.innerHTML = "Export data";
exportDataLink.classList.add('btn');
exportDataLink.download = current.toISOString() + '.exported.txt';
exportDataLink.style.position = "fixed";
exportDataLink.style.right = '5px';
exportDataLink.style.top = 0;
exportDataLink.style.border = "1px solid #888";
exportDataLink.style.boxShadow = "3px 3px #888";
exportDataLink.onclick = exportData;
document.body.appendChild(exportDataLink);

data = {
    'title': null ,
    'article': null
}

var body = document.getElementsByTagName("body")[0];

body.onclick = function($e){
    var num = $e.target.getAttribute('node_number')
    var valid = $e.target.getAttribute('valid')

    if(!valid){
        console.log("Element is invalid")
        return
    }

    if(data.length > 2){
        data = data.splice(0, 2)
    }

    console.log("node:", $e.target)

    if(!data.title && !data.article){
        data.title = num;
        console.log("data:", data)
        $e.target.classList.add('selected')
    }else if (data.title && !data.article){
        data.article = num;
        console.log("data:", data)
        $e.target.classList.add('selected')
    }else{
        selected = Array.from(document.getElementsByClassName("selected"))
        selected.forEach(function(el){
            el.classList.remove('selected')
        })

        data.title = num;
        data.article = null;
        $e.target.classList.add('selected')

    }

    if(data.title && data.article){
        var obj = {
            url: url,
            text: body_html,
            title: data.title,
            article: data.article,
        }

        var datasetItem = JSON.stringify(obj)
        localStorage.setItem(url, datasetItem)
    }
}
