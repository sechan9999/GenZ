# AWS Deployment Guide - Clinical Workflow Assistant

**Version**: 2.0.0
**Last Updated**: 2025-11-22

---

## Overview

This guide walks through deploying the Gen Z Clinical Workflow Assistant on AWS infrastructure with HIPAA compliance.

### AWS Services Used

- **Amazon EKS**: Kubernetes cluster hosting
- **Amazon RDS**: PostgreSQL database (metadata)
- **Amazon ElastiCache**: Redis caching
- **Amazon S3**: Encrypted data storage (backups, logs)
- **AWS Secrets Manager**: Secure credential management
- **Amazon CloudWatch**: Monitoring and logging
- **AWS WAF**: Web Application Firewall
- **AWS Shield**: DDoS protection
- **VPC**: Network isolation

---

## Architecture Diagram

```
                          ┌──────────────────┐
                          │   Route 53 (DNS)  │
                          └─────────┬────────┘
                                    │
                          ┌─────────▼────────┐
                          │  CloudFront      │
                          │  (CDN + WAF)     │
                          └─────────┬────────┘
                                    │
┌───────────────────────────────────┼───────────────────────────────────┐
│                                   │                                   │
│  VPC (10.0.0.0/16) - Healthcare-AI Region                           │
│                                   │                                   │
│  ┌────────────────────────────────┼────────────────────────────────┐ │
│  │  Public Subnets (10.0.1.0/24, 10.0.2.0/24)                     │ │
│  │                                │                                 │ │
│  │  ┌──────────────┐   ┌──────────▼───────────┐  ┌──────────────┐ │ │
│  │  │   NAT GW 1   │   │   Application LB     │  │   NAT GW 2   │ │ │
│  │  └──────┬───────┘   └──────────┬───────────┘  └──────┬───────┘ │ │
│  └─────────┼───────────────────────┼───────────────────────┼─────────┘ │
│            │                       │                       │             │
│  ┌─────────┼───────────────────────┼───────────────────────┼─────────┐ │
│  │  Private Subnets (10.0.10.0/24, 10.0.11.0/24)          │         │ │
│  │         │                       │                       │         │ │
│  │  ┌──────▼───────────────────────▼───────────────────────▼──────┐ │ │
│  │  │                  Amazon EKS Cluster                        │ │ │
│  │  │                                                             │ │ │
│  │  │  ┌──────────────────────────────────────────────────────┐ │ │ │
│  │  │  │  Gen Z Clinical Workflow Assistant Pods (3-10)      │ │ │ │
│  │  │  │  - RAG System (ChromaDB)                            │ │ │ │
│  │  │  │  - Guardrails Engine                                │ │ │ │
│  │  │  │  - HITL System                                      │ │ │ │
│  │  │  └──────────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────┬──────────────────────────┬───────────────┘ │
│                         │                          │                   │
│  ┌──────────────────────┼──────────────────────────┼───────────────┐ │
│  │  Database Subnets    │                          │               │ │
│  │                      │                          │               │ │
│  │  ┌──────────────────▼────────────┐  ┌──────────▼────────────┐ │ │
│  │  │  RDS PostgreSQL (Multi-AZ)    │  │  ElastiCache Redis    │ │ │
│  │  │  - Encrypted at rest          │  │  - Cluster mode       │ │ │
│  │  │  - Auto backup (7-day)        │  │  - Encryption         │ │ │
│  │  └───────────────────────────────┘  └───────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  AWS Services (External to VPC)                                 │ │
│  │                                                                   │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │ │
│  │  │ S3 (Encrypted) │  │ Secrets Manager│  │  CloudWatch Logs │ │ │
│  │  │ - Backups      │  │ - API Keys     │  │  - Audit Logs    │ │ │
│  │  │ - Audit Logs   │  │ - Credentials  │  │  - App Logs      │ │ │
│  │  └────────────────┘  └────────────────┘  └──────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Tools

```bash
# AWS CLI
aws --version  # >= 2.x

# kubectl
kubectl version --client  # >= 1.25

# eksctl
eksctl version  # >= 0.150

# Terraform (optional, for IaC)
terraform version  # >= 1.5

# Docker
docker --version  # >= 20.x
```

### AWS Account Setup

1. **AWS Account**: With admin access or appropriate IAM permissions
2. **AWS Region**: Choose HIPAA-eligible region (us-east-1, us-west-2, etc.)
3. **Cost Estimation**: ~$500-1000/month for production deployment

---

## Step-by-Step Deployment

### Step 1: Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# Set default region
export AWS_REGION=us-east-1

# Verify configuration
aws sts get-caller-identity
```

---

### Step 2: Create VPC and Networking

```bash
# Create VPC
aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=healthcare-ai-vpc},{Key=HIPAA,Value=true}]'

# Get VPC ID
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=healthcare-ai-vpc" --query "Vpcs[0].VpcId" --output text)

# Create public subnets (for ALB)
aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 \
  --availability-zone ${AWS_REGION}a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=public-subnet-1},{Key=Type,Value=Public}]'

aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.2.0/24 \
  --availability-zone ${AWS_REGION}b \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=public-subnet-2},{Key=Type,Value=Public}]'

# Create private subnets (for EKS nodes)
aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.10.0/24 \
  --availability-zone ${AWS_REGION}a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=private-subnet-1},{Key=Type,Value=Private}]'

aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.11.0/24 \
  --availability-zone ${AWS_REGION}b \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=private-subnet-2},{Key=Type,Value=Private}]'

# Create Internet Gateway
aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=healthcare-ai-igw}]'

IGW_ID=$(aws ec2 describe-internet-gateways --filters "Name=tag:Name,Values=healthcare-ai-igw" --query "InternetGateways[0].InternetGatewayId" --output text)

# Attach IGW to VPC
aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID

# Create NAT Gateways (for private subnet egress)
# Allocate Elastic IPs
aws ec2 allocate-address --domain vpc --tag-specifications 'ResourceType=elastic-ip,Tags=[{Key=Name,Value=nat-eip-1}]'
aws ec2 allocate-address --domain vpc --tag-specifications 'ResourceType=elastic-ip,Tags=[{Key=Name,Value=nat-eip-2}]'

# Get public subnet IDs and EIP allocation IDs
PUBLIC_SUBNET_1=$(aws ec2 describe-subnets --filters "Name=tag:Name,Values=public-subnet-1" --query "Subnets[0].SubnetId" --output text)
PUBLIC_SUBNET_2=$(aws ec2 describe-subnets --filters "Name=tag:Name,Values=public-subnet-2" --query "Subnets[0].SubnetId" --output text)
EIP_1=$(aws ec2 describe-addresses --filters "Name=tag:Name,Values=nat-eip-1" --query "Addresses[0].AllocationId" --output text)
EIP_2=$(aws ec2 describe-addresses --filters "Name=tag:Name,Values=nat-eip-2" --query "Addresses[0].AllocationId" --output text)

# Create NAT Gateways
aws ec2 create-nat-gateway --subnet-id $PUBLIC_SUBNET_1 --allocation-id $EIP_1 --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=nat-gw-1}]'
aws ec2 create-nat-gateway --subnet-id $PUBLIC_SUBNET_2 --allocation-id $EIP_2 --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=nat-gw-2}]'

# Configure route tables (simplified - complete via console or Terraform)
```

---

### Step 3: Create EKS Cluster

Using `eksctl` (recommended):

```bash
# Create EKS cluster configuration
cat > eks-cluster.yaml <<EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: healthcare-ai-cluster
  region: ${AWS_REGION}
  version: "1.28"
  tags:
    Environment: production
    HIPAA: "true"
    Compliance: healthcare

# VPC configuration
vpc:
  id: "${VPC_ID}"
  subnets:
    private:
      private-subnet-1: { id: "${PRIVATE_SUBNET_1}" }
      private-subnet-2: { id: "${PRIVATE_SUBNET_2}" }

# Managed node groups
managedNodeGroups:
  - name: genz-clinical-nodes
    instanceType: m5.2xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    volumeSize: 100
    volumeType: gp3
    volumeEncrypted: true
    privateNetworking: true
    labels:
      role: clinical-ai
      workload: healthcare
    tags:
      HIPAA: "true"
      k8s.io/cluster-autoscaler/enabled: "true"
      k8s.io/cluster-autoscaler/healthcare-ai-cluster: "owned"
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true
        albIngress: true

# CloudWatch logging
cloudWatch:
  clusterLogging:
    enableTypes: ["*"]
    logRetentionInDays: 30

# Encryption
secretsEncryption:
  keyARN: arn:aws:kms:${AWS_REGION}:ACCOUNT_ID:key/KMS_KEY_ID
EOF

# Create cluster
eksctl create cluster -f eks-cluster.yaml

# Update kubeconfig
aws eks update-kubeconfig --region ${AWS_REGION} --name healthcare-ai-cluster

# Verify
kubectl get nodes
```

---

### Step 4: Create RDS PostgreSQL Database

```bash
# Create DB subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name healthcare-ai-db-subnet \
  --db-subnet-group-description "Subnet group for healthcare AI database" \
  --subnet-ids $PRIVATE_SUBNET_1 $PRIVATE_SUBNET_2

# Create security group for RDS
aws ec2 create-security-group \
  --group-name healthcare-ai-rds-sg \
  --description "Security group for RDS PostgreSQL" \
  --vpc-id $VPC_ID

RDS_SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=healthcare-ai-rds-sg" --query "SecurityGroups[0].GroupId" --output text)

# Allow PostgreSQL access from EKS nodes
aws ec2 authorize-security-group-ingress \
  --group-id $RDS_SG_ID \
  --protocol tcp \
  --port 5432 \
  --cidr 10.0.0.0/16

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier healthcare-ai-postgres \
  --db-instance-class db.r5.xlarge \
  --engine postgres \
  --engine-version 15.3 \
  --master-username dbadmin \
  --master-user-password 'SECURE_PASSWORD_HERE' \
  --allocated-storage 100 \
  --storage-type gp3 \
  --storage-encrypted \
  --kms-key-id arn:aws:kms:${AWS_REGION}:ACCOUNT_ID:key/KMS_KEY_ID \
  --vpc-security-group-ids $RDS_SG_ID \
  --db-subnet-group-name healthcare-ai-db-subnet \
  --multi-az \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "Mon:04:00-Mon:05:00" \
  --enable-cloudwatch-logs-exports '["postgresql"]' \
  --tags Key=HIPAA,Value=true Key=Environment,Value=production

# Get DB endpoint
aws rds describe-db-instances \
  --db-instance-identifier healthcare-ai-postgres \
  --query "DBInstances[0].Endpoint.Address" \
  --output text
```

---

### Step 5: Create ElastiCache Redis

```bash
# Create cache subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name healthcare-ai-cache-subnet \
  --cache-subnet-group-description "Subnet group for healthcare AI cache" \
  --subnet-ids $PRIVATE_SUBNET_1 $PRIVATE_SUBNET_2

# Create security group for Redis
aws ec2 create-security-group \
  --group-name healthcare-ai-redis-sg \
  --description "Security group for ElastiCache Redis" \
  --vpc-id $VPC_ID

REDIS_SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=healthcare-ai-redis-sg" --query "SecurityGroups[0].GroupId" --output text)

# Allow Redis access from EKS nodes
aws ec2 authorize-security-group-ingress \
  --group-id $REDIS_SG_ID \
  --protocol tcp \
  --port 6379 \
  --cidr 10.0.0.0/16

# Create Redis replication group
aws elasticache create-replication-group \
  --replication-group-id healthcare-ai-redis \
  --replication-group-description "Redis cache for healthcare AI" \
  --engine redis \
  --cache-node-type cache.r5.large \
  --num-cache-clusters 2 \
  --automatic-failover-enabled \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --auth-token "SECURE_TOKEN_HERE" \
  --cache-subnet-group-name healthcare-ai-cache-subnet \
  --security-group-ids $REDIS_SG_ID \
  --snapshot-retention-limit 5 \
  --snapshot-window "03:00-05:00" \
  --tags Key=HIPAA,Value=true Key=Environment,Value=production

# Get Redis endpoint
aws elasticache describe-replication-groups \
  --replication-group-id healthcare-ai-redis \
  --query "ReplicationGroups[0].NodeGroups[0].PrimaryEndpoint.Address" \
  --output text
```

---

### Step 6: Configure AWS Secrets Manager

```bash
# Create secret for Anthropic API key
aws secretsmanager create-secret \
  --name healthcare-ai/anthropic-api-key \
  --description "Anthropic Claude API key for clinical AI" \
  --secret-string "sk-ant-api03-YOUR_KEY_HERE" \
  --kms-key-id arn:aws:kms:${AWS_REGION}:ACCOUNT_ID:key/KMS_KEY_ID \
  --tags Key=HIPAA,Value=true

# Create secret for PHI encryption key
aws secretsmanager create-secret \
  --name healthcare-ai/phi-encryption-key \
  --description "PHI encryption key for HIPAA compliance" \
  --secret-string "$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" \
  --kms-key-id arn:aws:kms:${AWS_REGION}:ACCOUNT_ID:key/KMS_KEY_ID \
  --tags Key=HIPAA,Value=true

# Create secret for database password
aws secretsmanager create-secret \
  --name healthcare-ai/postgres-password \
  --description "PostgreSQL database password" \
  --secret-string "SECURE_DB_PASSWORD" \
  --kms-key-id arn:aws:kms:${AWS_REGION}:ACCOUNT_ID:key/KMS_KEY_ID \
  --tags Key=HIPAA,Value=true

# Create secret for Redis auth token
aws secretsmanager create-secret \
  --name healthcare-ai/redis-auth-token \
  --description "Redis authentication token" \
  --secret-string "SECURE_REDIS_TOKEN" \
  --kms-key-id arn:aws:kms:${AWS_REGION}:ACCOUNT_ID:key/KMS_KEY_ID \
  --tags Key=HIPAA,Value=true
```

---

### Step 7: Install Kubernetes Secrets Store CSI Driver

```bash
# Install Secrets Store CSI Driver
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/secrets-store-csi-driver/v1.3.4/deploy/rbac-secretproviderclass.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/secrets-store-csi-driver/v1.3.4/deploy/csidriver.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/secrets-store-csi-driver/v1.3.4/deploy/secrets-store.csi.x-k8s.io_secretproviderclasses.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/secrets-store-csi-driver/v1.3.4/deploy/secrets-store.csi.x-k8s.io_secretproviderclasspodstatuses.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/secrets-store-csi-driver/v1.3.4/deploy/secrets-store-csi-driver.yaml

# Install AWS Secrets Manager provider
kubectl apply -f https://raw.githubusercontent.com/aws/secrets-store-csi-driver-provider-aws/main/deployment/aws-provider-installer.yaml

# Verify
kubectl get pods -n kube-system | grep secrets-store
```

---

### Step 8: Deploy Application to EKS

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/resources.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

# Verify deployment
kubectl get pods -n healthcare-ai
kubectl get svc -n healthcare-ai

# Check logs
kubectl logs -f deployment/genz-clinical-assistant -n healthcare-ai
```

---

### Step 9: Configure Application Load Balancer

```bash
# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=healthcare-ai-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Create Ingress
cat > ingress.yaml <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: genz-clinical-ingress
  namespace: healthcare-ai
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:REGION:ACCOUNT:certificate/CERT_ID
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS-1-2-2017-01
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
    alb.ingress.kubernetes.io/actions.ssl-redirect: '{"Type": "redirect", "RedirectConfig": { "Protocol": "HTTPS", "Port": "443", "StatusCode": "HTTP_301"}}'
spec:
  rules:
  - host: clinical-ai.yourhospital.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: genz-clinical-service
            port:
              number: 80
EOF

kubectl apply -f ingress.yaml

# Get ALB DNS name
kubectl get ingress -n healthcare-ai
```

---

### Step 10: Configure CloudWatch Monitoring

```bash
# Install CloudWatch Container Insights
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cloudwatch-namespace.yaml

kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-serviceaccount.yaml

# Configure CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-daemonset.yaml

# Verify
kubectl get pods -n amazon-cloudwatch
```

---

## Security Hardening

### Enable VPC Flow Logs

```bash
aws ec2 create-flow-logs \
  --resource-type VPC \
  --resource-ids $VPC_ID \
  --traffic-type ALL \
  --log-destination-type cloud-watch-logs \
  --log-group-name /aws/vpc/healthcare-ai \
  --deliver-logs-permission-arn arn:aws:iam::ACCOUNT_ID:role/VPCFlowLogsRole
```

### Enable AWS Shield and WAF

```bash
# Enable AWS Shield Standard (free)
# Shield Advanced requires subscription ($3000/month)

# Create WAF WebACL
aws wafv2 create-web-acl \
  --name healthcare-ai-waf \
  --scope REGIONAL \
  --default-action Allow={} \
  --rules file://waf-rules.json \
  --visibility-config SampledRequestsEnabled=true,CloudWatchMetricsEnabled=true,MetricName=HealthcareAIWAF

# Associate with ALB
aws wafv2 associate-web-acl \
  --web-acl-arn arn:aws:wafv2:REGION:ACCOUNT:regional/webacl/healthcare-ai-waf/ID \
  --resource-arn arn:aws:elasticloadbalancing:REGION:ACCOUNT:loadbalancer/app/ALB_NAME/ID
```

---

## Cost Optimization

### Estimated Monthly Costs

| Service | Configuration | Est. Cost/Month |
|---------|---------------|-----------------|
| **EKS Cluster** | Control plane | $73 |
| **EC2 (Nodes)** | 3x m5.2xlarge | $450 |
| **RDS PostgreSQL** | db.r5.xlarge, Multi-AZ | $600 |
| **ElastiCache Redis** | 2x cache.r5.large | $200 |
| **Data Transfer** | 100GB/month | $50 |
| **CloudWatch Logs** | 50GB/month | $25 |
| **S3 Storage** | 500GB | $12 |
| **Secrets Manager** | 5 secrets | $2 |
| **ALB** | Application Load Balancer | $25 |
| **NAT Gateway** | 2x NAT GW | $90 |
| **EBS Volumes** | 500GB total | $50 |
| **Total** | | **~$1,577/month** |

### Cost Savings Tips

1. **Use Spot Instances** for non-critical workloads (save 70%)
2. **Enable Auto Scaling** to reduce nodes during off-hours
3. **Use S3 Intelligent-Tiering** for logs and backups
4. **Reserve Instances** for stable workloads (save 40%)

---

## HIPAA Compliance Checklist

- ✅ **Encryption at Rest**: All EBS, RDS, S3, ElastiCache encrypted
- ✅ **Encryption in Transit**: TLS 1.2+ for all communications
- ✅ **Audit Logging**: CloudWatch Logs, VPC Flow Logs, RDS logs
- ✅ **Access Control**: IAM roles, security groups, NACLs
- ✅ **Data Backup**: Automated RDS backups (7 days), S3 versioning
- ✅ **Network Isolation**: Private subnets, no direct internet access
- ✅ **Secrets Management**: AWS Secrets Manager with KMS encryption
- ✅ **Monitoring**: CloudWatch alarms, Container Insights
- ✅ **Business Associate Agreement (BAA)**: Sign AWS BAA for HIPAA compliance

---

## Troubleshooting

### Common Issues

**Issue**: Pods not starting
```bash
kubectl describe pod POD_NAME -n healthcare-ai
kubectl logs POD_NAME -n healthcare-ai
```

**Issue**: Cannot connect to RDS
```bash
# Check security group
aws ec2 describe-security-groups --group-ids $RDS_SG_ID

# Test connection from pod
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- psql -h RDS_ENDPOINT -U dbadmin -d postgres
```

**Issue**: High costs
```bash
# Check Cost Explorer
aws ce get-cost-and-usage --time-period Start=2025-11-01,End=2025-11-30 --granularity MONTHLY --metrics BlendedCost
```

---

## Maintenance

### Backup and Restore

```bash
# Backup FHIR data to S3
kubectl exec -n healthcare-ai POD_NAME -- tar czf - /app/gen_z_agent/healthcare | \
  aws s3 cp - s3://healthcare-ai-backups/$(date +%Y%m%d)/healthcare-data.tar.gz

# Restore from S3
aws s3 cp s3://healthcare-ai-backups/20251122/healthcare-data.tar.gz - | \
  kubectl exec -i -n healthcare-ai POD_NAME -- tar xzf - -C /app/gen_z_agent/
```

### Updates

```bash
# Update EKS cluster
eksctl upgrade cluster --name healthcare-ai-cluster --approve

# Update application
kubectl set image deployment/genz-clinical-assistant genz-clinical=genz-clinical-assistant:2.1.0 -n healthcare-ai

# Rollback if needed
kubectl rollout undo deployment/genz-clinical-assistant -n healthcare-ai
```

---

## Support

- **AWS Support**: Enterprise Support (recommended for HIPAA)
- **Documentation**: https://docs.aws.amazon.com/
- **HIPAA on AWS**: https://aws.amazon.com/compliance/hipaa-compliance/

---

**Last Updated**: 2025-11-22
**Version**: 2.0.0
