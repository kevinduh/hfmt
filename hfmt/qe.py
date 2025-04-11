from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-cometkiwi-da")#"Unbabel/wmt20-comet-qe-da")
MODEL = load_from_checkpoint(model_path)

def run_example():
	data = [
    	{
        	"src": "The output signal provides constant sync so the display never glitches.",
        	"mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
    	},
    	{
        	"src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
        	"mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
    	},
    	{
        	"src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
        	"mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
    	}
	]
	model_output = MODEL.predict(data, batch_size=8, gpus=1)
	print (model_output)
	# output: Prediction([('scores', [0.3048405051231384, 0.23435941338539124, 0.6128205060958862]), ('system_score', 0.384006808201472)])
	print(model_output.system_score)

def get_qe_score(refs, hyps):	
	data = [{"src": ref, "mt": hyp} for ref, hyp in zip(refs, hyps)]
	model_output = MODEL.predict(data, batch_size=8, gpus=1)
	return model_output.system_score


