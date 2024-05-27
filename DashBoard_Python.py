from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import dash_daq as daq
import dash_bootstrap_components as dbc          # pip install dash-bootstrap-components
import pandas as pd
import numpy as np
import pyarrow
import plotly.graph_objects as go
import json
import plotly.figure_factory as ff
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
#from sklearn.model_selection import train_test_split

#SEED_NUMBER = 1234
pd.set_option('display.max_columns', None)

t_data_full = pd.read_parquet('/Users/andreapiemontese/Desktop/2Round.nosync/parquet_input_files/t_data_full.parquet')

t_data_train=pd.read_parquet('/Users/andreapiemontese/Desktop/2Round.nosync/parquet_input_files/t_data_train.parquet')
t_data_test=pd.read_parquet('/Users/andreapiemontese/Desktop/2Round.nosync/parquet_input_files/t_data_test.parquet')

#t_data_train, t_data_test = train_test_split( t_data_full, test_size=0.2, random_state=SEED_NUMBER, stratify=t_data_full['fraud'])

t_Dim_merchant = pd.read_parquet('/Users/andreapiemontese/Desktop/2Round.nosync/parquet_input_files/t_Dim_merchant.parquet')
t_Dim_customer = pd.read_parquet('/Users/andreapiemontese/Desktop/2Round.nosync/parquet_input_files/t_Dim_customer.parquet')


y_test=pd.read_csv("/Users/andreapiemontese/Desktop/2Round.nosync/model_predictions.csv")['Y_Test']
y_test_pred=pd.read_csv("/Users/andreapiemontese/Desktop/2Round.nosync/model_predictions.csv")['Y_Test_Predicted']
y_test_pred_proba=pd.read_csv("/Users/andreapiemontese/Desktop/2Round.nosync/model_predictions.csv")['Y_Test_Pred_Proba']

feature_importance=pd.read_csv("/Users/andreapiemontese/Desktop/2Round.nosync/feature_importance.csv").set_index(['Unnamed: 0'])


def cleaning(df):
    #Does not seem necessary so far
    return df

def CrossTabulation(data):
    age_crosstab = pd.crosstab(data['age'], data['fraud'], normalize='index') * 100
    gender_crosstab = pd.crosstab(data['gender'], data['fraud'], normalize='index') * 100
    industry_crosstab = pd.crosstab(data['industry'], data['fraud'], normalize='index') * 100
    return [age_crosstab.round(2).rename(columns={False:'No Fraud (%)',True:'Fraud(%)'}).sort_values(['Fraud(%)']).iloc[:, ::-1],
            gender_crosstab.round(2).rename(columns={False:'No Fraud (%)',True:'Fraud(%)'}).sort_values(['Fraud(%)']).iloc[:, ::-1],
            industry_crosstab.round(2).rename(columns={False:'No Fraud (%)',True:'Fraud(%)'}).sort_values(['Fraud(%)']).iloc[:, ::-1]
            ]
    
    
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.layout = dbc.Container([
      html.Br(),
    html.H1("Analysis of Transaction Data", className='mb-2', style={'textAlign':'left'}),
    html.Hr(style = {'size' : '50', 'borderColor':'stonegray','borderHeight': "10vh", "width": "95%",}),
    dbc.Card(
        dbc.CardBody([
            dbc.Row([dbc.Col( [ html.H2("I. Summarised Statistics on the Dataset", className='mb-2', style={'textAlign':'left'}),
                    
             ])]),
            html.Hr(),
            html.H3("I.I Descriptives & Conditional Visualization", className='mb-2', style={'textAlign':'left'}),
            dbc.Row([html.P(""" Target variable 'fraud' is binary. Amongst the factor variables only one is numeric, the rest
                            is either categorical or ordinal. Within the training set only 1.22% of transactions are fraudulent (sample imbalance).""")], 
                            style={'font-size':'20px'},className='mt-4'),
            dbc.Row([html.P(""" Transaction Amount undoubtedly relevant by looking at its empirical distribution conditional on the 'fraud' flag.
                            Some industry display a fraud rate suggesting systematic frauds occur within them, as opposed to some others, i.e. 
                            booking vacation v. grocery shopping by credit card. Merchant label is definitely informative, in line with industry and
                            has indeed 20% correlation with the latter. Age & gender of the customer seem to be least informative.""",
                            style={'font-size':'20px'})] ,className='mt-4'),
                    dbc.Row([html.P(""" In training set 5769 frauds out of 472492 transactions (1.22% ca). In test set 
                                    1431 frauds out of 118123 transactions (1.21% ca)""",
                            style={'font-size':'20px'})] ,className='mt-4'),
            dbc.Row([
                
                    dbc.Spinner( dcc.Graph(id='Distribution') )
                ]),
                            
            dbc.Row([
                
                dbc.Col([dash_table.DataTable(id = 'Summary_Statistics',
                                       style_header={'backgroundColor': 'rgb(50, 50, 50)',
                                                        'color': 'white'},
                                        style_data={'backgroundColor': 'rgba(0, 0, 0, 0)',
                                                        'color': 'white' },
                                     style_as_list_view=True,fill_width=True)],width=8),
                    
                dbc.Col([
                    
                    dcc.Dropdown(
                        [*t_data_train.columns],
                        'fraud', 
                                 id='select_categorical')
                    ],width=4)
                    
                
                    
                

            ],justify='left'),
            html.Hr(),
            html.H3("I.II Joint Distribution of Fraud and Factor Variables", className='mb-2', style={'textAlign':'left'}),
            html.Br(),
            
            dbc.Row([html.P(""" While in the case of categorical variables the contingency table convey same information, in 
                            the case of age now we are looking at the fraud v. non-fraud distribution conditional on age group,
                            i.e. other way around compared to visualization.""")], 
                            style={'font-size':'20px'},className='mt-4'),
            dbc.Row([html.P(""" The most relevant label appears to be industry of the merchant, in line with theoretical intution and
                            preliminary visualization analysis.""",
                            style={'font-size':'20px'})] ,className='mt-4'),
            
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(CrossTabulation(t_data_train)[0].reset_index().to_dict('records'),
                                             style_header={'backgroundColor': 'rgb(50, 50, 50)',
                                                            'color': 'white'},
                                             style_data={'backgroundColor': 'rgba(0, 0, 0, 0)',
                                                            'color': 'white'},
                                         style_as_list_view=True,fill_width=False)
                    ],width=4),
                    dbc.Col([
                    dash_table.DataTable(CrossTabulation(t_data_train)[1].reset_index().to_dict('records'),
                                             style_header={'backgroundColor': 'rgb(50 50, 50)',
                                                            'color': 'white'},
                                             style_data={'backgroundColor': 'rgba(0, 0, 0, 0)',
                                                            'color': 'white'},
                                         style_as_list_view=True,fill_width=False)
                    ],width=4),
                        dbc.Col([
                    dash_table.DataTable(CrossTabulation(t_data_train)[2].reset_index().to_dict('records'),
                                             style_header={'backgroundColor': 'rgb(50, 50, 50)',
                                                            'color': 'white'},
                                             style_data={'backgroundColor': 'rgba(0, 0, 0, 0)',
                                                            'color': 'white'},
                                         style_as_list_view=True,fill_width=False)
                    ],width=4)
                ]),
  


        ])
    ),
    
    dbc.Card(
        dbc.CardBody([
             dbc.Row([dbc.Col( [ html.H2("II. Geographical Analysis", className='mb-2', style={'textAlign':'left'}),
                                
            
             ])]), html.Hr(),
            
              html.Br(),
            
             dbc.Row([ 
                 
                 dbc.Col([
                     dbc.Spinner( dbc.Row(dcc.Graph(id='MapMerch_Trans')) )
                     ],width=7),
                 dbc.Col([
                            dash_table.DataTable( id='Merch_Community' ,
                                       style_header={'backgroundColor': 'rgb(50, 50, 50)',
                                                        'color': 'white'},
                                        style_data={'backgroundColor': 'rgba(0, 0, 0, 0)',
                                                        'color': 'white' },
                                     style_as_list_view=True,fill_width=False) ]
                     ,width=5)
                     ]),
                 
          
             html.Br(),
             dbc.Row([
                 dbc.Col([
                     html.H3('Towns Combination with the Most Transactions: ')
                     ],width=6),
                 dbc.Col([
                 dash_table.DataTable( id='Merch_Consumer_Community' ,
                                       style_header={'backgroundColor': 'rgb(50, 50, 50)',
                                                        'color': 'white'},
                                        style_data={'backgroundColor': 'rgba(0, 0, 0, 0)',
                                                        'color': 'white' },
                                     style_as_list_view=True,fill_width=True)
                 ],width=6)
                 ]),
             dbc.Row([html.P(""" Grouping merchants by community (town), one can check the size of the 
                             group-specific dataset. The merchant community with highest number of transactions is
                             Bussigny, with lowest Renes. Towns combination with highest number of transactions between
                             seller and buyer is reported below.""")], 
                            style={'font-size':'20px'},className='mt-4'),
 
            ])
        
        ),
    
    
    dbc.Card(
        dbc.CardBody([
              dbc.Row([dbc.Col( [ html.H2("III. Model-based analysis of Fraud Frequency", className='mb-2', style={'textAlign':'left'}),
                                
            
             ])]), html.Hr(),
              
              dbc.Row([
                  dbc.Col([
                      dbc.Spinner( dcc.Graph(id='confusion-matrix') )
                      ],width=7),
                  dbc.Col([
                      html.Br(),html.Br(),
                       dash_table.DataTable( id='metrics1' ,
                                       style_header={'backgroundColor': 'rgb(50, 50, 50)',
                                                        'color': 'white'},
                                        style_data={'backgroundColor': 'rgba(0, 0, 0, 0)',
                                                        'color': 'white' },
                                     style_as_list_view=True,fill_width=True),html.Br(),html.Br(),
                        dash_table.DataTable( id='metrics2' ,
                                       style_header={'backgroundColor': 'rgb(50, 50, 50)',
                                                        'color': 'white'},
                                        style_data={'backgroundColor': 'rgba(0, 0, 0, 0)',
                                                        'color': 'white' },
                                     style_as_list_view=True,fill_width=True),html.Br(),
                        html.Div([
                        html.Div([html.P('Marchant Out')],
                                 style={'width':'30%','display':'inline-block',
                        'margin': '0px 0px 0px 0px','font-weight':'bold',
                        'font-size':'16px'}),
                        html.Div([daq.ToggleSwitch(id='merch_in',value=False,size=80)],
                                 style={'width':'25%','display':'inline-block'}),
                        html.Div([html.P('Marchant In')],
                                 style={'width':'30%','display':'inline-block',
                                        'margin': '0px 0px 0px 0px','font-weight':'bold',
                                        'font-size':'16px'})  ],className='row')
                      ],width=5)
                  ]),
              dbc.Row([html.P(""" In sample accuracy is 99%, as well as out of sample. However,
                              given the large imbalance between the support of fraud versus non-fraud 
                              instances, it is better to look at precision and recall for True and False 
                              cases separately, and then aggregate with no weighted average""")], 
                            style={'font-size':'20px'},className='mt-4'),
            dbc.Row([html.P(""" First Logistic Regression model is with no merchant information, to 
                            focus on transaction granularity information as 'counterparty rating' 
                            effect is obvious from seller fraud's descriptives and visualization.
                            However, by including merchant information model performance out-of-sample
                            remains comparable (around 95%) in terms of precision and significantly improves
                            in terms of recall (from 79% to 86%).""",
                            style={'font-size':'20px'})] ,className='mt-4'),
              
              dbc.Row([
                  dbc.Col([ 
                      dbc.Spinner(dcc.Graph(id='AUC'))
                      ],width=5),
                  dbc.Col([
                      dbc.Spinner(dcc.Graph(id='Feature_Importance'))
                      ],width=7)
                  
                  ])
              
            ])
        )
    
    ])

@app.callback(
    Output('Distribution','figure'),
    Output('Summary_Statistics', 'data'),
    Input('select_categorical','value')
)
def Question1(variable,df=t_data_train):
    if variable in ['gender','industry','merchant','customer']:
        grouped_df = df.groupby([variable, 'fraud']).size().reset_index(name='count')
        grouped_df['percentage'] = grouped_df['count'] / grouped_df.groupby(variable)['count'].transform('sum')* 100
        
        fig = px.bar(grouped_df,
             x=variable, y='percentage', color='fraud',color_discrete_sequence=['teal','indianred'],
             barmode='stack',#labels={'payer_gender': 'Payer Gender', 'count': 'Count'},
             title=f'{variable} Distribution by Fraud')
    elif variable=='fraud':
        fig=px.pie(df, names='fraud', title='Proportion of Fraud vs. Non-Fraud',
             labels={'fraud': 'Fraud Status'},color_discrete_sequence=['teal','indianred'])
    else:
        df=df.sort_values([variable])
        fig=px.histogram(df, x=variable, marginal='box',
                 color='fraud',histnorm='percent',
                 color_discrete_sequence=['teal','indianred'],
                 )
    fig.update_layout( template='plotly_dark', plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)' )
    
    df=df[[variable]].describe().round(2).T.reset_index()
    df = df[['index'] + [col for col in df.columns if col != 'index']]
    return  fig, df.to_dict('records')


@app.callback(
    Output('MapMerch_Trans','figure'),
    Output('Merch_Community','data'),
    Input('select_categorical','value')
)
def Question2_part1(ciao,df1=t_data_train,df2=t_Dim_merchant):
    df=pd.merge(df1,df2, how='left',left_on='merchant',right_on='merchant' )
    #prova=pd.DataFrame(df.groupby(['Plz4']).size()).rename(columns={0:'Number_of_Transactions'})
    prova=pd.DataFrame(df.groupby(['Plz4','Town']).size()).rename(columns=
                                                                        {0:'Number_of_Transactions'}).sort_values(['Town'])
    prova=pd.merge(prova.reset_index(),prova.groupby(['Town']).sum(),how='left',left_on='Town',right_on='Town').set_index(['Plz4'])[['Town','Number_of_Transactions_y']]
    prova=prova.rename(columns={'Number_of_Transactions_y':'N Transactions'})
    with open("/Users/andreapiemontese/Desktop/2Round.nosync/ch-plz.geojson", 'r') as file:
        data = json.load(file)
        data['features']=[ data['features'][i] for i in [*np.nonzero([np.isin(i['id'],[*t_Dim_merchant.Plz4]) for i in data['features']])[0]] ]
    fig = px.choropleth_mapbox(prova, geojson=data, 
                           locations=[*prova.index],
                           featureidkey ='id',
                           color_continuous_scale='teal',
                           color=prova['N Transactions'],
                           mapbox_style = 'carto-positron',
                           zoom=10.2,
                           opacity=0.5,
                           center = {"lat": 46.519962,
                                             "lon": 6.633597 },hover_data=["Town"]
                          )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        template='plotly_dark',paper_bgcolor= 'rgba(0, 0, 0, 0)' )
    
    
    
    df=pd.DataFrame(df.groupby(['Town']).size()).rename(columns={0:'Number_of_Transactions'}).sort_values(['Number_of_Transactions'])
    return fig,df.reset_index().to_dict('records')

@app.callback(
    Output('Merch_Consumer_Community','data'),
    Input('select_categorical','value')
    )
def Question2_part2(ciao,df1=t_data_train,df2=t_Dim_merchant,df3=t_Dim_customer):
    df=pd.merge(df1,df2, how='left',
            left_on='merchant',right_on='merchant').merge(df3, how='left',
                                                         left_on='customer',right_on='customer')
    df=pd.DataFrame( pd.DataFrame(df.groupby(['Town_x','Town_y']).size()).reset_index().rename(
    columns={'Town_x':'Merchant Town',
             'Town_y':'Customer Town',
             0:'Number of Transactions'}).sort_values('Number of Transactions').iloc[-1,:] ).T
    return df.to_dict('records') 


@app.callback(
    Output('confusion-matrix','figure'),
    Output('metrics1','data'),
    Output('metrics2','data'),
    Output('AUC','figure'),
    Output('Feature_Importance','figure'),
    Input('merch_in','value')
    )
def Question3(merch_in,Y_test=y_test,Y_test_pred=y_test_pred, Y_test_pred_proba=y_test_pred_proba, Feature_importance=feature_importance):
    if merch_in==True:
        Y_test=pd.read_csv("/Users/andreapiemontese/Desktop/2Round.nosync/model_predictions_1.csv")['Y_Test']
        Y_test_pred=pd.read_csv("/Users/andreapiemontese/Desktop/2Round.nosync/model_predictions_1.csv")['Y_Test_Predicted']
        Y_test_pred_proba=pd.read_csv("/Users/andreapiemontese/Desktop/2Round.nosync/model_predictions_1.csv")['Y_Test_Pred_Proba']
        
        Feature_importance=pd.read_csv("/Users/andreapiemontese/Desktop/2Round.nosync/feature_importance_1.csv").set_index(['Unnamed: 0'])
        
    conf_matrix = confusion_matrix(Y_test, Y_test_pred)
    labels = ['Non-Fraud', 'Fraud']
    z = conf_matrix
    x = labels
    y = labels
    
    fig_conf_matrix = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=conf_matrix, colorscale='teal'
    )
    fig_conf_matrix.update_layout(title_text='Confusion Matrix',
                                  xaxis=dict(title='Predicted Label'),
                                  yaxis=dict(title='True Label'),
                                  plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                  paper_bgcolor= 'rgba(0, 0, 0, 0)',template='plotly_dark' )
   
    d1=pd.DataFrame(classification_report(Y_test, Y_test_pred,output_dict=True)).iloc[:2,:2].reset_index()
    d2=pd.DataFrame(pd.DataFrame(classification_report(Y_test, Y_test_pred,output_dict=True)).iloc[:2,][['macro avg']]
                       ).reset_index().round(2)
    print(pd.DataFrame(classification_report(Y_test, Y_test_pred,output_dict=True)))
    fpr, tpr, _ = roc_curve(Y_test, Y_test_pred_proba)
    roc_auc = roc_auc_score(Y_test, Y_test_pred_proba)
    
    # Create the ROC curve plot
    fig = go.Figure()
    
    # Add the ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                             line=dict(color='teal', width=2), 
                             name='ROC curve (area = %0.2f)' % roc_auc))
    
    # Add the diagonal line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                             line=dict(color='navy', width=2, dash='dash'), 
                             showlegend=False))
    
    # Update the layout
    fig.update_layout(
        xaxis=dict(title='False Positive Rate', range=[0.0, 1.0]),
        yaxis=dict(title='True Positive Rate', range=[0.0, 1.05]),
        title='Receiver Operating Characteristic',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        template='plotly_dark' 
    )
    
    
    #feature_importance = pd.DataFrame(best_logreg_model.coef_[0], index=X_train.columns, columns=['importance']).sort_values(by='importance', ascending=False)
    
  
    # Create the feature importance bar plot
    fig1 = go.Figure()
    
    # Add the bars
    fig1.add_trace(go.Bar(
        x=Feature_importance.importance,
        y=Feature_importance.index,
        orientation='h',
        marker=dict(color=feature_importance.importance, colorscale='Teal')
    ))
    
    # Update the layout
    fig1.update_layout(
        title='Feature Importance based on Logistic Regression Coefficients',
        xaxis=dict(title='Importance'),
        yaxis=dict(title='Feature'),
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        template='plotly_dark' 
    )

    #print(f"ROC-AUC Score: {roc_auc_score(y_test, y_test_pred_proba)}")
    return fig_conf_matrix, d1.round(2).to_dict('records'),d2.to_dict('records'),fig, fig1

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

  #No NaN as per such encoded value -- Check for non-numeric values in numeric




