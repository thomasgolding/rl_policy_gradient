#!/usr/bin/env bash


# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.

image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

account=$(aws sts get-caller-identity --query Account --output text)


## check if aws-call went ok
if [ $? -ne 0 ]
then
    exit 255
fi


region=$(aws configure get region)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"


# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1



if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

$(aws ecr get-login --region ${region} --no-include-email)


docker build  -t ${image} --file Dockerfile_aws .
docker tag ${image} ${fullname}

docker push ${fullname}


