import requests
import xml.etree.ElementTree as ET
import numpy as np

#%%
def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",

                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        print('Found', numFound)
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            print('Docs:', d)
            if verbose:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            # all_files.append(d['url'])
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return np.array(sorted(all_files))


#%%
# facets= 'mip_era,activity_id,model_cohort,product,source_id,institution_id,source_type,nominal_resolution,experiment_id,sub_experiment_id,variant_label,grid_label,table_id,frequency,realm,variable_id,cf_standard_name,data_node'
#%%
result = esgf_search(
    # activity_id='CMIP,ScenarioMIP',
    experiment_id='historical',
    frequency='day',
    variable_id='mrro',
    source_id="ACCESS-CM2",
    member_id="r1i1p1f1"
# files_type='HTTPServer'
    # facets=facets,
)

print(result)
print(len(result))
