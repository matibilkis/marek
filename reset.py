import shutil

def remove(folder):
    try:
        sure = input("Are you sure you wanna remove: "+i + " ? (y/n): ")
        if sure == "y":
            shutil.rmtree(folder)
        else:
            "not removed, some error, maybe directory doesnt't exist"
    except Exception:
        pass

folders = ["1L2PH0.1R", "dicts", "figures"]
for i in folders:

        remove(i)
