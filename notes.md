in GI_bleeding:

Transfusion_ids in mimic_iii: 
in d_items.csv the labels include "PRBC" or "Packed RBC"
[  5649, 5751,  30179,   7597,  30104,  42324,  42588,  30001, 30004,  42239,
   46407,  46612,  46124,  45750,  42740,42186, 227070, 226368]

Two recording systems in mimic iii --> carevue and metavision mimiciv only metavision

inputevents_cv (carevue) 2001-2008
inputsevents_mv (Metavision) 2008-2012+

in MIMIC-IV we only have metavision as inputevents.csv
in mimiciv d_items we only have [227070, 226368] --> inputevents.itemid in transfusion_codes -->5268 unique admissions
in mimic iii we have 14 codes --> combine inputevents_mv and cv --> 12734 unique admissions (1759 for [227070, 226368])

So in MIMICIV we have less data