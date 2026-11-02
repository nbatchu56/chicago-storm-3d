const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1280,height:720}});
await p.goto('file://'+process.cwd()+'/'+process.argv[2],{waitUntil:'networkidle',timeout:60000});await p.waitForTimeout(1500);
await p.screenshot({path:process.argv[3]});await b.close();})();
