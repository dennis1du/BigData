library(MASS)
library(tilting)
library(glmnet)
library(foreach)
library(doParallel)

##Total number of training examples and test examples, n
n = 1e+5

##Number of features 
p = 500

##Covariance matrix of the mulvariate normal distribution
I_cov = diag(1,nrow = p)

##Generate X
x_center = rep(0,p)
x_ori = mvrnorm(n,x_center,I_cov)

##Calculate teh number of non-zero entries of the beta vector
num_non_0 = ceiling(p/20)

##Choose ground true beta
beta_T = matrix(0,nrow = p,ncol = 1)
beta_T[1:num_non_0] = runif(num_non_0,-0.5,0.5)

##Generate the true response y, y = x*beta+N(0,sigma), N(0,sigma) represents random noise
sigma = 0.1
epsilon = rnorm(n,0,sigma)
y_T = x_ori%*%beta_T+epsilon

##Set initial predictors, the initial betas not necessary to be sparse vector
# beta0 = matrix(0,nrow = p,ncol = 1)
# beta0_nonzeros = sample(1:p,num_non_0)
# beta0[beta0_nonzeros] = runif(num_non_0,-0.5,0.5)

beta0 = runif(p,-0.5,0.5)

##Define the objective function
obj_f = function(x,y,beta,lamda)
{
  obj_f = 1/n*((norm(y-x%*%beta,type = '2'))^2+lamda*sum(abs(beta)))
  return(obj_f)
}

##Define a sign function of beta
sign_beta = function(beta)
{
  beta = sign(beta)
  if (is.element(0,beta))
  {
    index_0 = which(beta==0)
    beta[index_0] = runif(length(index_0),-1,1)
  }
  return(beta)
}

##Define the subgradient function
sub_g = function(x,y,beta,lamda)
{
  sub_g = (-2*(t(x)%*%(y-x%*%beta))+lamda*sign_beta(beta))/n
  return(sub_g)
}

##Define the proximal function

proximal = function(x,y,beta,lamda,alpha,ite)
{
  # obj_fxn = c()
  for (i in 1:ite)
  {
    beta = beta+2/length(y)*alpha*(t(x)%*%(y-x%*%beta))
    beta = apply(cbind(beta-lamda*alpha,rep(0,length(beta))), 1, max)-apply(cbind(-beta-lamda*alpha,rep(0,length(beta))), 1, max)
    # obj_fxn = c(obj_fxn,obj_f(x,y,beta,lamda))
  }
  # return(obj_fxn)
  return(beta)
}

##Cross validation to pick lamda (use proximal)
##List of lamda's being selected
lamda_cv = c(0,5e-4,1e-3,1.5e-3,2e-3,2.5e-3,5e-3,1e-2,5e-2,0.1)
length(lamda_cv)
##Number of folds to break the original data set
k_fold = 5
# ##Break the data set into k_fold group
# x_break = split(sample(n,n,replace = FALSE),as.factor(1:k_fold))

##Combine the x and y of the original data set
xy_ori = cbind(x_ori,y_T)
dim(xy_ori)

##Define alpha parameter
alpha = 0.1
ite_cv = 500

# se_cv = c()
# mean_cv = c()
mean_beta_cv = matrix(0,nrow = p,ncol = length(lamda_cv))
beta_cv_t = c()
##generate random 5-fold data set
index_values = sample(1:k_fold,size = n,replace = TRUE)


#########################################################################

t_start = Sys.time()
cv = c()
beta_cv = matrix(0,nrow = p,ncol = k_fold)
pro_cvin = txtProgressBar(1,k_fold,style = 3)
for (k in 1:k_fold)
{
  setTxtProgressBar(pro_cvin,k)
  index_out = which(index_values==k)
  left_out_data = xy_ori[index_out,]
  left_in_data = xy_ori[-index_out,]
  beta_cv[,k] = proximal(left_in_data[,-dim(left_in_data)[2]],left_in_data[,dim(left_in_data)[2]],beta0,lamda_cv[5],alpha,ite_cv)
  y_out_data = left_out_data[,dim(left_out_data)[2]]
  x_out_data = left_out_data[,-dim(left_out_data)[2]]
  cv_temp = 1/length(y_out_data)*sum((y_out_data-x_out_data%*%beta_cv[,k])^2)
  cv = c(cv,cv_temp)
}
t_end = Sys.time()
t_cal = t_end-t_start

#########################################################################


mean_cv = c(mean_cv,mean(cv))
se_cv = c(se_cv,1/sqrt(k_fold)*sd(cv))
beta_cv_t = c(beta_cv_t,rowMeans(beta_cv))

mean_beta_cv = matrix(beta_cv_t,nrow = p,ncol = length(lamda_cv),byrow = FALSE)
##Calculate the column 2 norm of the difference between each estimated beta and the true beta
diff_beta_cv = col.norm(mean_beta_cv-rep(beta_T,length(lamda_cv)))
diff_beta_cv


##Plot the one standard deviation of cv vs lamda
plot(lamda_cv[1:7],mean_cv[1:7],ylim = range(c(mean_cv[1:7]-se_cv[1:7],mean_cv[1:7]+se_cv[1:7])),type = 'o',pch = 15,col = 'red',xlab = 'lamda',ylab = 'Mean +/- SD')
abline(h = mean_cv[4]+se_cv[4])
arrows(0.001,0.01,0.001,0.01003,length = 0.25)
text(0.001,0.01,expression("0.001"))

##plot the 2 norm of each beta-beta_T
plot(lamda_cv[1:7],diff_beta_cv[1:7],type = 'o',col = 'red',pch = 15,xlab = 'Lambda',ylab = '2-norm of Beta-Beta_T')


##Pick the optimal lamda, lamda*
##lamda* = 0.001  or 0.005
lamda_star = 0.005

###########################################
##part (a)
###########################################
##Constant alpha
alpha = 0.01
lamda = lamda_star
##ite below controls how many iterations will be done in the update of beta's
ite = 1000
beta_temp = beta0
obj_f_con = c()
pro_con = txtProgressBar(1,ite,style = 3)
for (i in 1:ite)
{
  setTxtProgressBar(pro_con,i)
  beta_temp = beta_temp-alpha*sub_g(x_ori,y_T,beta_temp,lamda)
  obj_f_con = c(obj_f_con,obj_f(x_ori,y_T,beta_temp,lamda))
  beta_a_con = beta_temp
}

##Decreasing alpha
a = 0.01
beta_temp = beta0
obj_f_des = c()
for (i in 1:ite)
{
  setTxtProgressBar(pro_con,i)
  alpha = a/sqrt(i)
  beta_temp = beta_temp-alpha*sub_g(x_ori,y_T,beta_temp,lamda)
  obj_f_des = c(obj_f_des,obj_f(x_ori,y_T,beta_temp,lamda))
  beta_a_des = beta_temp
}

##Visualization
plot(1:ite,obj_f_des,ylim = c(0,50),type = 'o',col = 'blue',pch = 16,xlab = 'Number of Iterations',ylab = 'Objective Function Value')
lines(1:ite,obj_f_con,type = 'o',col = 'red',pch = 15)
lines(1:ite,obj_f_pro,type = 'o',col = 'Black',pch = 1)
lines(1:ite,obj_f_line,type = 'o',col = 'cyan',pch = 2)
legend('topright',c('Constant alpha','Decreasing alpha','Proximal Method','Backtrack Line Search'),col = c('red','blue','Black','cyan'),pch = c(15:16,1,2),cex = 1,bty = 0)
# legend('topright',c('Constant alpha','Decreasing alpha'),col = c('red','blue'),pch = c(15:16),cex = 1,bty = 0)

###########################################
##part (b)-proximal method
###########################################
##USe the following for-loop is different from using the function define above: proximal()
#######################################
alpha_p = 0.01
lamda_p = lamda_star
beta_temp = beta0
obj_f_pro = c()
# t_start_pro = Sys.time()
# for (i in 1:ite)
# {
#   setTxtProgressBar(pro_con,i)
#   beta_temp = beta_temp+2/length(y_T)*alpha*(t(x_ori)%*%(y_T-x_ori%*%beta_temp))
#   beta_temp = apply(cbind(beta_temp-lamda_p*alpha_p,rep(0,length(beta_temp))), 1, max)-apply(cbind(-beta_temp-lamda_p*alpha_p,rep(0,length(beta_temp))), 1, max)
#   obj_f_pro = c(obj_f_pro,obj_f(x_ori,y_T,beta_temp,lamda_p))
#   beta_b_pro = beta_temp
# }
# t_end_pro = Sys.time()
# t_cal_pro = t_end_pro-t_start_pro
obj_f_pro = proximal(x_ori,y_T,beta0,lamda_p,alpha_p,ite)

##Visualization
plot(1:ite,obj_f_pro,type = 'l',col = 'red',pch = 15,xlab = 'Number of Iterations',ylab = 'Objective Function Value',main = 'Proximal Method')

###########################################
##part (c) --- Line search
###########################################
alpha0 = 0.01
c_alpha = 0.8
ite = 1000
lamda = 0.005
beta_last = beta0
v_line = beta0
M_last = beta0
alpha = alpha0
obj_f_line = c()
L = 1
alpha_min = min(alpha0,c_alpha/L)
pro_line = txtProgressBar(1,ite,style = 3)
##Calculate the 1st step
for (i in 1:ite)
{
  setTxtProgressBar(pro_line,i)
  alpha = alpha
  beta_line = M_last+2/n*alpha*t(x_ori)%*%(y_T-x_ori%*%M_last)
  beta_line = apply(cbind(beta_line-lamda*alpha,rep(0,length(beta_line))), 1, max)-apply(cbind(-beta_line-lamda*alpha,rep(0,length(beta_line))), 1, max)
  v_line = beta_last+(i+1)/2*(beta_line-beta_last)
  M_line = (i/(i+2))*beta_line+2/(i+2)*v_line
  ##Condition to continue decrease alpha
  cond = (norm((y_T-x_ori%*%beta_line),type = '2'))^2-(norm((y_T-x_ori%*%M_line),type = '2'))^2
  +2*(t(x_ori)%*%(y_T-x_ori%*%M_line))%*%(beta_line-M_line)-0.5/alpha*(norm(beta_line-M_line,type = '2'))^2
  if (cond>0)
  {
    alpha = max(alpha*c_alpha,alpha_min)
  }
  beta_last = beta_line
  M_last = M_line
  obj_f_line = c(obj_f_line,obj_f(x_ori,y_T,beta_line,lamda))
}

plot(1:ite,obj_f_line,type = 'l',col = 'red',xlab = 'Number of Iterations',ylab = 'Objective Function Value',main = 'Backtracking Line Search')

###########################################
##part (d)
###########################################
glm_fit = glmnet(x_ori,y_T,family = 'gaussian',alpha = 1,lambda = lamda_cv)
summary(glm_fit)
beta_glmfit = glm_fit$beta
lamda_glmfit = glm_fit$lambda
obj_f_glmfit = obj_f(x_ori,y_T,beta_glmfit,lamda_glmfit)
obj_f_glmfit
beta_glmfit