var http = require("http");
var fs = require("fs");
var url = require("url");

var root_cwd = process.cwd();  //current working dir

//current working dir files
var cwd = {'path':'',	'files':[]} 

function get_view(URL){
	URL = URL.split("=");
	cur_view = URL[URL.length-1];
	
	if(cur_view == "" || URL.length < 2) view = {'name':"main.html", 'path':root_cwd+"\\"+cur_view};
	else view = {'path':root_cwd+"\\"+cur_view, 'name':cur_view};
	
	return view;
}

//create server
http.createServer(function (request, response){
	
	//parse request containing filename
	onfocus = get_view(request.url);
	
	var pathname = view.path;
	
	//print pathname
	console.log("requesting for: "+pathname);
	
	console.log(pathname);
	
	//read requsted from filesystem
	fs.readFile(pathname, function (err, data){
		if(err){
			console.log(err);
	
			//send http header
			//http status 404: page not found
			//context-type:text/plain
			response.writeHead(404, {'Content-Type':'text/plain'});
		}else{
			
			//send http header
			//http status 200:ok
			//context-type:text/plain
			
			response.writeHead(200, {'Content-Type':'text/html'});
			
			//write to output
			response.write(data.toString());
			
		}
		
		//send response
		response.end('');
	});
}).listen(5000);
