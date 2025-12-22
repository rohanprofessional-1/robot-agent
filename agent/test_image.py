from roboflow import Roboflow
rf = Roboflow(api_key="h0igL7eBCHMWZ88AWKja")
workspaces = rf.workspace("researchworker").project("tube-detection-x52mi-9pgxe").version(1).model

print(workspaces)