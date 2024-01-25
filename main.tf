# We strongly recommend using the required_providers block to set the
# Azure Provider source and version being used
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=3.0.2"
    }
  }
}

# Local variables
locals {
  image_name = "ttca_prompt_test"
  image_tag  = "v0.21"

}
# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {}
}

# Create a resource group
resource "azurerm_resource_group" "rg" {
  name     = "ttca"
  location = "uksouth"
}

# Container registry
resource "azurerm_container_registry" "acr" {
  name                = "ttcacontainerRegistry"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

resource "null_resource" "push_docker_container" {
  triggers = {
    image_tag  = local.image_tag
    image_name = local.image_name
  }

  provisioner "local-exec" {
    command = <<-EOT
            az acr build --no-logs --image ${local.image_name}:${local.image_tag} --registry ${azurerm_container_registry.acr.login_server} --file Dockerfile .
            EOT
  }
}



# Service plan
resource "azurerm_service_plan" "asp" {
  name                = "ttcaprompttestasp"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  os_type             = "Linux"
  sku_name            = "B3"
}

# App Service
resource "azurerm_app_service" "aas" {
  name                = "ttcaprompttestas"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  app_service_plan_id = azurerm_service_plan.asp.id

  site_config {
    linux_fx_version = "DOCKER|${azurerm_container_registry.acr.login_server}/${local.image_name}:${local.image_tag}"
    always_on        = "true"
  }

  app_settings = {
    DOCKER_REGISTRY_SERVER_URL      = azurerm_container_registry.acr.login_server
    DOCKER_REGISTRY_SERVER_USERNAME = azurerm_container_registry.acr.admin_username
    DOCKER_REGISTRY_SERVER_PASSWORD = azurerm_container_registry.acr.admin_password
    WEBSITES_PORT                   = 8501
  }
}
