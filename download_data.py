from roboflow import Roboflow
rf = Roboflow(api_key="z6t12rm0MrYPIWreQ0kW")
project = rf.workspace("signlinga").project("sign-language-xpq5z")
version = project.version(1)
dataset = version.download("voc")
                
                