import requests

# Define the base URL of the PubChem PUG REST API
url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Define the search term for sterols
search_term = "sterol"

# Define the list of properties to retrieve for each compound
properties = ["IUPACName", "Synonym", "MolecularFormula", "MolecularWeight"]

# Send the API request to search for sterols by IUPAC name
search_url = f"{url}/compound/name/{search_term}/cids/JSON"
response = requests.get(search_url)
data = response.json()

# Extract the list of CIDs from the response
cids = data["IdentifierList"]["CID"]

# Create a new text file to store the list of sterols
with open("/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/sterols.txt", "w") as file:
    # Loop through each CID and retrieve its properties
    for cid in cids:
        # Send the API request to retrieve the properties for the current CID
        props_url = f"{url}/compound/cid/{cid}/property/{','.join(properties)}/JSON"
        response = requests.get(props_url)
        data = response.json()

        # Check if the response contains a PropertyTable key
        if "PropertyTable" not in data:
            print(f"No properties found for CID {cid}")
            continue

        # Extract the properties from the response
        iupac_name = data["PropertyTable"]["Properties"][0]["IUPACName"]
        synonyms = data["PropertyTable"]["Properties"][0]["Synonym"]
        formula = data["PropertyTable"]["Properties"][0]["MolecularFormula"]
        mw = data["PropertyTable"]["Properties"][0]["MolecularWeight"]

        # Write the properties to the file
        file.write(f"CID: {cid}\n")
        file.write(f"IUPAC Name: {iupac_name}\n")
        if synonyms:
            file.write(f"Synonyms: {', '.join(synonyms)}\n")
        file.write(f"Molecular Formula: {formula}\n")
        file.write(f"Molecular Weight: {mw}\n")
        file.write("\n")

        # Print the JSON response for debugging
        print(f"Properties for CID {cid}:\n{data}")
