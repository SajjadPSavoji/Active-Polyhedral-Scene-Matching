### A simple webform sumbmitter for the polyhedral data set
import mechanize
from tqdm import tqdm
N = 10 # number of instances
for _ in tqdm(range(N)):
    url = "https://polyhedral.eecs.yorku.ca/"
    br = mechanize.Browser()
    br.set_handle_robots(False) # ignore robots
    br.open(url)
    br.select_form(name="polyForm")
    br['data_type'] = ["generator"]        # "generator", "upload"
    br["light"] = ["fixed"]                # "fixed", "homogenous"
    br["numberObjects"] = "1"              # or any other number in [1, 180]
    br["layout"] = ["separate",]           # "touching" or "intersecting"
    br["numberViews"] = "3"                # number of views to be generated
    br["emailAddress"] = "__YOUR_Gmail__" # your email address

    res = br.submit()
    content = res.read()