%computes derivatives w/rt weights for BP
function  [dWL_cum,dW_Lminus1_cum,delta_L_cum,delta_Lminus1_cum] = compute_dW_from_sensitivities(Wji,bj_vec,phi_code1,Wkj,bk_vec,phi_code2,training_patterns,targets)

  [K,P] =size(targets); %dim of output vec and num training patterns
  [J,I] = size(Wji); %input vector dim I and num interneurons, J
  
  delta_L_cum = zeros(K,1);
  delta_L = delta_L_cum; %set same size
  delta_Lminus1_cum = zeros(J,1);
  delta_Lminus1 = delta_Lminus1_cum; %init size
  dWL_cum = Wkj*0; 
  dWL = Wkj*0;
  dW_Lminus1 = zeros(J-1,I);
  dW_Lminus1_cum = Wji*0;
  
  
      %need to compute outputs of both j and k layers for all stimulus
      %patterns
      [outputs_j,outputs_k]=eval_2layer_fdfwdnet(Wji,bj_vec,phi_code1,Wkj,bk_vec,phi_code2,training_patterns);
      err_vecs = outputs_k - targets;
      phi_prime_L_vecs = fnc_phi_prime(phi_code2,outputs_k); %make sure this is consistent w/ act. fnc.
  
      deltas_L = (phi_prime_L_vecs).*(err_vecs); %FIXed!!! compute delta_L for every pattern excitation
  
      delta_L_cum= sum(deltas_L,2); %net delta_L is sum of all columns of deltas_L
      
             
      %compute remaining delta_l(n) recursively:
      phi_prime_Lminus1_vecs = fnc_phi_prime(phi_code1,outputs_j);%outputs_j.*(1-outputs_j); 
      for p=1:P
        deltas_Lminus1(:,p) = (transpose(Wkj)*deltas_L(:,p)).*(phi_prime_Lminus1_vecs(:,p)); %FIXed!!! put in recursive relationship
      end  
  
      delta_Lminus1_cum= sum(deltas_Lminus1,2); %net bias sensitivities for this layer; add all columns
      
      %given all deltas(n) for all layers, can compute synapse sensitivities
      dW_Lminus1 = (deltas_Lminus1(:,1))*transpose(training_patterns(:,1)); %FIXed!!!
      dW_Lminus1_cum = dW_Lminus1;    
      dWL = deltas_L(:,1).*transpose(outputs_j(:,1)); %FIXed!!!
      dWL_cum = dWL;
      for p=2:P
          %layer L synapse sensitivities:
          dWL = (deltas_L(:,p)).*transpose(outputs_j(:,P)); %FIXed!! 
          dWL_cum = dWL_cum + dWL;
          %layer L-1 synapse sensitivities
         dW_Lminus1 = (deltas_Lminus1(:,p)).*transpose(training_patterns(:,p)); %FIXed!!        
         dW_Lminus1_cum = dW_Lminus1_cum+dW_Lminus1; 
         %could make this a loop for arbitrary number of layers...
      end
  
  