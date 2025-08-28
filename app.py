import streamlit as st
from Summary_recon import run_reconciliation

st.title("Bank Reconciliation Tool")

batch_file = st.file_uploader("Upload 1881 Batch File", type=["xlsx"])
rta_file = st.file_uploader("Upload RTA File", type=["xlsx"])

if st.button("Run Reconciliation"):
    if batch_file and rta_file:
        output_file = run_reconciliation(batch_file, rta_file, "Reconciliation_Output.xlsx")
        st.success("Reconciliation Completed!")
        with open(output_file, "rb") as f:
            st.download_button("Download Reconciliation Report", f, file_name=output_file)
    else:
        st.error("Please upload both files.")
