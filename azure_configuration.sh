#!/bin/bash

# Set your Azure subscription and resource group information
subscription_id="UAO"  # Replace with your actual subscription ID
resource_group="ML_FARO_PROJECT"
location="westus"  # Use "westus" for US West region

# Set up Azure Machine Learning workspace
aml_workspace_name="your_ml_workspace"  # Replace with your desired name
aml_workspace_region="westus"

# Set up Azure Kubernetes Service (AKS)
aks_cluster_name="faro"
aks_node_count=2

# Log in to Azure
az login

# Set the default subscription
az account set --subscription $subscription_id

# Create a resource group
az group create --name $resource_group --location $location

# Create Azure Machine Learning workspace
az ml workspace create -w $aml_workspace_name -g $resource_group -l $aml_workspace_region

# Get the Azure Machine Learning workspace information
aml_workspace_id=$(az ml workspace show -w $aml_workspace_name -g $resource_group --query id -o tsv)

# Create Azure Kubernetes Service (AKS)
az aks create \
  --resource-group $resource_group \
  --name $aks_cluster_name \
  --node-count $aks_node_count \
  --enable-addons monitoring \
  --location $location \
  --generate-ssh-keys

# Get AKS credentials
az aks get-credentials --resource-group $resource_group --name $aks_cluster_name

# Install Azure ML extension for AKS
az ml extension install -w $aml_workspace_id --cluster-name $aks_cluster_name --resource-group $resource_group --ignore-errors

# Verify the Azure ML extension is installed
az ml extension list -w $aml_workspace_id --cluster-name $aks_cluster_name --resource-group $resource_group

# Deploy your application to AKS using Azure ML
# (This step will depend on your specific application and deployment requirements)

echo "Script execution completed."
