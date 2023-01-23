#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Feature importance
feature_importances = grid_search.best_estimator_.feature_importances_
X_train1 = pd.DataFrame(X_train1)
fi = feature_importances  
fi_df = pd.DataFrame({'feature': list(X_train1.columns), 'feature importance': fi[:]}).sort_values('feature importance', ascending = False)
fi_df
sns.barplot(fi_df['feature importance'],fi_df['feature'] ,orient = "h", color = 'gray')
plt.savefig("featureimportanceRF.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')
fi_df.to_csv("RF_fi.csv")

#Change the variable names (for example, _Day 1 to (POD1)) on Excel
#RF_fi to RF_fi2

fi_df2 = pd.read_csv('RF_fi2.csv')
plt.figure(figsize=(20, 17))
sns.set(font_scale = 3)
sns.set_style("ticks")
sns.barplot(fi_df2['feature importance'],fi_df2['feature'] ,orient = "h", color = 'gray')
plt.savefig("featureimportance.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')

