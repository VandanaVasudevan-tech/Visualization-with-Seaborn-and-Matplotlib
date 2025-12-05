import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('tested.csv')

plt.figure(figsize=(8, 5))
sns.countplot(x='Embarked', hue='Sex', data=df)
plt.title("Passenger Count by Embarked Port (Separated by Sex)")
plt.xlabel("Embarked Port")
plt.ylabel("Number of Passengers")
plt.show()
# Observation: Most passengers boarded from 'S'. Overall, there were more males than females.

plt.figure(figsize=(10, 5))
sns.swarmplot(x='Pclass', y='Age', hue='Sex', data=df)
plt.title("Age Distribution Across Passenger Classes (Categorized by Sex)")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.show()
# Observation: First-class passengers tend to be older, while third-class has more young passengers.
# Gender distribution is fairly balanced.

gender_counts = df['Sex'].value_counts()
labels = gender_counts.index
sizes = gender_counts.values
explode = [0.1, 0] if labels[0] == 'female' else [0, 0.1]
plt.figure(figsize=(7, 7))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    explode=explode,
    shadow=True,
    startangle=90
)
plt.title("Gender Distribution of Passengers")
plt.show()
# Observation: Males (~63.6%) outnumber females (~36.4%) on the Titanic.

cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Survived']
corr = df[cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    linewidths=0.5,
    square=True
)
plt.title("Correlation Heatmap of Titanic Features")
plt.show()
# Observation: Fare negatively correlates with Pclass; higher fare slightly increases survival chances.

plt.figure(figsize=(10, 6))
sns.violinplot(
    x='Pclass',
    y='Fare',
    hue='Sex',
    data=df,
    split=True,
    palette='Set2'
)
plt.title("Fare Distribution by Class and Gender")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()
# Observation: First-class passengers paid much higher fares. Overlap exists for fares in 2nd and 3rd class.

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Age',
    y='Fare',
    hue='Survived',
    data=df,
    palette='Set1'
)
plt.title("Scatter Plot: Age vs Fare (Colored by Survival Status)")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title='Survived')
plt.show()
# Observation: Survivors tend to have paid higher fares. Age alone is not a strong predictor of survival.
