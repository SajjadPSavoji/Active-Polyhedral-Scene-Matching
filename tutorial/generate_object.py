### A simple webform sumbmitter for 
import mechanize

url = "https://polyhedral.eecs.yorku.ca/"
br = mechanize.Browser()
br.set_handle_robots(False) # ignore robots
br.open(url)
br.select_form(name="polyForm")
br['data_type'] = ["generator"] # "generator", "upload"
br["light"] = ["fixed"] # "fixed", "homogenous"
br["numberObjects"] = "1" # or any
br["layout"] = ["separate",] # "touching" or "intersecting"
br["numberViews"] = "3"
br["emailAddress"] = "Savoji@yorku.ca"

res = br.submit()
content = res.read()