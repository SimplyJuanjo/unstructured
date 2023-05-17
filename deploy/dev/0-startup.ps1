# Get credentials
az account set --subscription 53348303-e009-4241-9ac7-a8e4465ece27
az aks get-credentials --resource-group raitogpt2 --name fileLoaderDev

#kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.6.1/cert-manager.yaml
#kubectl apply --validate=false -f https://github.com/jetstack/cert-manager/releases/download/v1.0.2/cert-manager.crds.yaml
#kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.1.0/deploy/static/provider/cloud/deploy.yaml
#kubectl apply -f ./clusterissuer.yaml
kubectl apply -f ./updatedcertificate.yaml
#kubectl apply -f ./fileuploader-ingress.yaml
#kubectl apply -f ./secrets-dev.yaml
#kubectl apply -f ./fileuploader-deployment.yaml
#kubectl apply -f ./fileuploader-service.yaml

# Show cluster
kubectl cluster-info

# Show services
kubectl get services

#verify deployment
kubectl get deployments

# View pods
kubectl get pods

