########## REGULÄRE AUSDRÜCKE
import re


### Match-Objekt Methoden
# group()	Return the string matched by the RE
# start()	Return the starting position of the match
# end()	    Return the ending position of the match
# span()	Return a tuple containing the (start, end) positions of the match

### Zeichenauswahl
# [ab] = Zeichen aus Auswahl (hier a oder b)
# [^ab] = ausser angegebene Zeichen - Komplement (hier alls ausser a oder b)
#   => ^ in [] also andere Bedeutung!!!


### Zeichenklassen (gross = negation)
# \d = Zahl = [0-9]
# => Negation: \D = keine Zahl = [^0-9] = []
# \w = Buchstabe,Ziffer,Unterstrich = [a-zA-Z0-9_]
# \s = Leerzeichen

### Spezielle Zeichen
# ^ = Stringanfang
# $ = Stringende
# [^0-9] = Komplement (hier: keine zahl)
#

#### Quantoren
# Der voranstehende Ausdruck...
# {min,max} = mindestens min-mal und maximal max-mal
# ? = optional = {0,1}
# + = mindestens einmal {1,}
# * = darf beliebig oft {0,}
# {n} = n-mal = {n,n}



#### Kompilieren
# Reguläre Expression durch kompilieren zu Pattern-Obj für Operationen wie suchen/ersetzen
# (r für raw -> kein problem mit backslashes)
p = re.compile(r'ab+',re.IGNORECASE)


#### Matching
## Matching des Anfangs des Strings - prüft Anfang(!) des Strings
match = p.match('abbbccc') # kein match => None

## Finden des Pattern an beliebiger Stelle im String mit search
p.search('babbbccc')
print(match)
if match:
    print(match.group()) # => abbb
else:
    print("No Match")

## Alle matches finden
for match in p.finditer('abbbcabcc'):
    print("Match: ",match.group())

## Splitting
p.split('cabbbd', maxsplit=0) # ['c','d']

## Ersetzen
# () für matchgroup (kann auch schon im ersten string für wiederholung benutzt werden)
re.sub(r'(\+49)(800|\d{4})(\d+)',r'\1 \2 \3','+494131423')  # +49 4131 423

## Aus komplexen match mit Variablen nach Dictonary
format_pat= re.compile(
    r"(?P<host>[\d\.]+)\s"
    r"(?P<identity>\S*)\s"
    r"(?P<user>\S*)\s"
    r"\[(?P<time>.*?)\]\s"
    r'"(?P<request>.*?)"\s'
    r"(?P<status>\d+)\s"
    r"(?P<bytes>\S*)\s"
    r'"(?P<referer>.*?)"\s'
    r'"(?P<user_agent>.*?)"\s*')   # => wird zu einem String
logPath = "~/Documents/Data/access_log.txt"

URLCounts = {}
with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            split = request.split()
            if len(split) < 3:
                print(split)
                continue
            (action, URL, protocol) = split
            if URL in URLCounts:
                URLCounts[URL] = URLCounts[URL] + 1
            else:
                URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))
