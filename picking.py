from tcia_utils import nbia

# Define where you want your merged dataset to live
MASTER_FOLDER = "my_merged_ai_dataset/"

# 1. Grab 5 CT scans from a Lung Cancer collection
print("Fetching Lung CTs...")
lung_cts = nbia.getSeries(collection="NSCLC-Radiomics", modality="CT")
nbia.downloadSeries(lung_cts, number=5, path=MASTER_FOLDER)

# 2. Grab 5 MRI scans from a Brain Cancer collection
print("Fetching Brain MRIs...")
brain_mris = nbia.getSeries(collection="UPENN-GBM", modality="MR")
nbia.downloadSeries(brain_mris, number=5, path=MASTER_FOLDER)

safe_download(collection_name="MIDRC-RICORD-1c", modality_type="DX", num_scans=5)

# 3. Grab 5 X-Rays (CR/DX) from a Chest collection
print("Fetching Chest X-Rays...")
chest_xrays = nbia.getSeries(collection="MIDRC-RICORD-1a", modality="CR")
nbia.downloadSeries(chest_xrays, number=5, path=MASTER_FOLDER)

print("Downloads complete! All datasets merged into:", MASTER_FOLDER)