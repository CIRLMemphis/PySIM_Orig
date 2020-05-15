import urllib.request
username = 'user1'
password = '123456'
baseurl = 'https://login.microsoftonline.com/'
password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
password_mgr.add_password(None, baseurl, username, password)
handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
opener = urllib.request.build_opener(handler)
urllib.request.urlretrieve("https://livememphis-my.sharepoint.com/personal/cvan_memphis_edu/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fcvan%5Fmemphis%5Fedu%2FDocuments%2FCIRLData%2FSIM%5FNN%2Finput%2Ezip", "input.zip",auth=(username,password))
