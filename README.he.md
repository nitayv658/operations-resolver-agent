# 🤖 Operations Resolver Agent — סוכן תפעול אוטונומי

קווסט #4, חלק א' (Place-IL). סוכן יחיד שפותר פניות תמיכה של GlobalCart: קורא את
הפנייה, קורא לכלים כדי לשלוף את ההזמנה / הלקוח / המדיניות, מחליט על תוצאה
תפעולית (זיכוי אוטומטי / דחייה / הסלמה), ומחזיר פלט מובנה שאפשר לבקר.

> 🇬🇧 English version: [`README.md`](README.md) — same content, same structure.

מארז הכלים והדאטה יושבים ב-[`starter-kit/`](starter-kit/) ו**לא נגענו בהם** —
ראו [`starter-kit/README.md`](starter-kit/README.md) לכלים עצמם. כל מה שמתואר
כאן הוא הסוכן שנבנה *מסביב* למארז. המטלה המקורית שמורה ב-[`docs/quest-brief/`](docs/quest-brief/).

---

## 1. מה בנינו, ולמה זה סוכן ולא workflow

מדריך הרקע של הקווסט מבחין בין workflow (הנתיב קבוע בקוד) לבין agent (המודל
מחליט). כאן זה סוכן, וזו החלטה מודעת:

| | מה זה היה נראה | למה זה לא מספיק |
|---|---|---|
| **Workflow** | `get_order_details` → `get_user_profile` → `check_return_policy` → `process_refund`, תמיד באותו סדר | פנייה על הזמנה שלא קיימת לא צריכה בכלל להגיע ל-`process_refund`. הזמנה שטרם נשלחה נעצרת אחרי הכלי הראשון |
| **Agent** ✅ | המודל מחליט אילו מארבעת הכלים לקרוא, באיזה סדר, ומתי לעצור | פנייה בשפה טבעית לא מגלה מראש אם הבעיה במשלוח, בחיוב או במוצר |

**אין בקוד pipeline קשיח.** מקרה טיפוסי אכן מגיע לקרוא לכל ארבעת הכלים פחות או
יותר בסדר הזה — אבל זה המודל שמנמק את דרכו לשם דרך תיאורי הכלים, לא נתיב בקרה
שכתוב ב-`agent.py`.

## 2. 🏗️ הארכיטקטורה — ארבעה קבצים, שני עולמות

```
resolver_agent/
├── tool_loop.py    מנוע גנרי: send → tool_use → tool_result → send
├── output_tool.py  סכמת submit_resolution + ולידציה ואכיפה
├── prompts.py      ה-system prompt
└── agent.py        ResolverAgent — הקובץ היחיד שיודע שמדובר ב-GlobalCart
```

הפיצול הזה הוא ההחלטה המרכזית בפרויקט:

| | `tool_loop.py` | `agent.py` |
|---|---|---|
| **מה הוא יודע** | רק את המכניקה של לופ tool-calling | GlobalCart, זיכויים, ארבעת הכלים |
| **מה הוא מקבל** | `tool_schemas` + `tool_registry` כלשהם | — |
| **בחלק ב'** | נשאר **ללא שינוי**, כל סוכן בצוות משתמש בו | נכתב מחדש לכל סוכן |

`tool_loop.py:1-8` אומר את זה במפורש: הוא לא יודע כלום על זיכויים, על GlobalCart
או על צורת פלט. `agent.py:127-128` הוא התפר — שם נבנית רשימת הכלים (ארבעת הכלים
האמיתיים מ-`mock_services.py` + כלי חמישי, `submit_resolution`), נכנס
ה-`SYSTEM_PROMPT`, והתמליל הגולמי הופך לשלושת שדות הפלט הנדרשים.

> 🔧 **מבחן פשוט לכל שינוי עתידי:** אם הוספתם ל-`tool_loop.py` משהו שיודע מה זה
> זיכוי — הפרתם את החוזה. לוגיקה עסקית שייכת ל-`agent.py`.

## 3. למה לא פרימוורק?

התשובה הקצרה: **מה שפרימוורק בעיקר עושה כאן כבר נעשה, ומה שנשאר זה בדיוק המקום
שבו יושבים כל ה-guardrails הנבדקים.**

המארז שקיבלנו הוא framework-agnostic במכוון — ה-README שלו
(`starter-kit/README.md:180-186`) מספק מתאמים מוכנים ל-LangChain, CrewAI,
PydanticAI ו-OpenAI Tools. כלומר פרימוורק היה זמין לחלוטין; לא להשתמש בו זו
בחירה, לא אילוץ.

**מה פרימוורק היה תורם כאן בפועל:**

| מה פרימוורק נותן | האם היה עוזר כאן |
|---|---|
| תרגום פונקציות Python לסכמת כלים | ❌ מיותר. `TOOL_SCHEMAS` כבר בפורמט המדויק של Anthropic — ה-README של המארז אומר את זה מפורשות: *"TOOL_SCHEMAS is already in the right shape"* |
| הלופ עצמו | ⚠️ 202 שורות קוד בסך הכל (246 כולל תיעוד) |
| retry ו-backoff | ❌ ה-SDK של Anthropic כבר עושה את זה פנימית (`max_retries`, ברירת מחדל 2) |
| פלט מובנה | ❌ נפתר דרך כלי חמישי עם סכמה — ראו סעיף 5 |
| לוגים ו-observability | ⚠️ `logging_utils.py`, 78 שורות, ובדיוק בפורמט שרצינו |

**והסיבה המכריעה** — כל guardrail שהמטלה בודקת יושב *בתוך* הלופ:

| ה-guardrail | איפה | מה היה נדרש מול פרימוורק |
|---|---|---|
| כפיית `submit_resolution` בתור אחרון עם `tool_choice` | `tool_loop.py:228-240` | שליטה באיטרציה **האחרונה** ספציפית |
| סירוב לקריאה חוזרת עם אותם `(name, args)` | `tool_loop.py:173-188` | יירוט ה-dispatch והזרקת תוצאה סינתטית |
| חסימת מידע של לקוח אחר לפני שהוא מגיע למודל | `agent.py:60-103` | עטיפת ה-registry בגבול ה-dispatch |
| `ModelAPIError` שנושא את עקבות הכלים החלקיות | `tool_loop.py:49-64` | פרימוורקים עוטפים שגיאות API בטיפוסים משלהם ובדרך כלל מאבדים את התמליל החלקי |

ארבע מתוך המלכודות בטבלת המלכודות של מדריך הרקע (`max_iterations` + התנהגות
מוגדרת אחרי שגיאה, guardrail בקוד ולא בפרומפט, לופ אינסופי, פלט לא מובנה) נפתרות
בשורות בודדות במקום הנכון — במקום להיאבק באבסטרקציה.

> 💡 **הגינות אינטלקטואלית:** המטלה עצמה **ניטרלית** בנושא. `task.md` מפנה
> לסרטוני הדרכה "בפרימוורק שבחרתם" ומונה ארבעה שילובים אפשריים, ומדריך הרקע אפילו
> מציע structured output "של הפרימוורק" כפתרון למלכודת הפלט הלא מובנה. ההתייחסות
> הקרובה ביותר לגישה שלנו היא פריט #7 ברשימת הקריאה — *Building agents with the
> Claude Agent SDK*, המתויג "אם החלטתם לבנות את הלופ בעצמכם ולא דרך פרימוורק".
> זו רשות, לא המלצה. הנימוק שלמעלה עומד בזכות עצמו.

**האופציה שכן שווה להזכיר:** לא LangChain, אלא `client.beta.messages.tool_runner`
של ה-SDK של Anthropic עצמו — שמריץ את מחזור הבקשה→הרצה→לופ ואף חושף hooks לכל תור
(שערי אישור, יירוט תוצאות, `max_iterations`). הוא באמת היה יכול לארח את רוב
ה-guardrails שלמעלה. שתי סיבות שהוא עדיין לא מתאים כאן: הוא ב-**beta**, ותנאי
העצירה הכפוי (`tool_choice` נעול בתור האחרון) הוא התנהגות מותאמת לאיטרציה
אחרונה, לא hook מוכן.

### 💾 מטמון פרומפטים (Prompt caching)

ה-API של Anthropic הוא stateless — כל round-trip בתוך קריאת `resolve()` אחת
שולח מחדש את כל התמליל, וה-system prompt ורשימת חמשת הכלים זהים בכל round-trip
כזה (אף אחד מהם לא משתנה באמצע המקרה). `tool_loop.py` מסמן את שניהם עם
breakpoint מסוג `cache_control` של Anthropic (`_cacheable_system` /
`_cacheable_tools`), מחושב פעם אחת לכל קריאה ומשמש בכל תור, כולל התור האחרון
הכפוי — כך שתורות אחרי הראשון משתמשים ב-prefix השמור ב-cache במקום שהמודל
יעבד אותו מחדש מאפס. שתי הפונקציות בונות אובייקטים **חדשים** במקום לשנות את
`tool_schemas` של הקורא במקום — הרשימה הזו נבנית פעם אחת ב-`ResolverAgent.
__init__` ומשמשת מחדש בכל קריאת `resolve()` על אותו instance, כך ששינוי שלה
במקום היה מדליף breakpoint ישן (או גרוע מזה, state משותף) בין פניות לא
קשורות.

## 4. 🧰 הכלים — ארבעה מהמארז, ואחד משלנו

ארבעת הכלים הראשונים מגיעים מ-`starter-kit/mock_services.py` **ללא שינוי**, דרך
`TOOL_SCHEMAS` ו-`TOOL_REGISTRY`. הם מוצגים כאן בסדר שסוכן בדרך כלל משתמש בהם —
אבל שוב, זה הסדר האופייני שהמודל מגיע אליו, לא סדר שכתוב בקוד:

| # | הכלי | על מה הוא עונה | קלט | מה חוזר |
|---|---|---|---|---|
| 1 | `get_order_details` | מה הוזמן, מתי הגיע, ובאיזה מצב | `order_id` | סטטוס משלוח, תאריכי הזמנה ומסירה, סכום כולל, הפריטים והמצב שכל אחד הגיע בו, כתובת ואם שונתה אחרי ההזמנה |
| 2 | `get_user_profile` | מי הלקוח, מה הדרגה שלו, היסטוריית הזיכויים והסיכון | `user_id` | דרגה (VIP ⬅️ חלון החזרה ארוך יותר ותקרה גבוהה יותר), ותק, LTV, היסטוריית זיכויים, דגלי הונאה וציון הונאה |
| 3 | `check_return_policy` | האם התביעה עדיין זכאית, ולפי איזה סעיף | `order_id`, `reason` | `eligible` + `verdict` (`ELIGIBLE` / `OUTSIDE_RETURN_WINDOW` / `NON_RETURNABLE_CATEGORY` / `ORDER_NOT_REFUNDABLE`), `applicable_policies`, `auto_refund_cap_usd`, `max_refundable_amount`, `requires_escalation` + `escalation_reasons`, `return_window_days` / `days_since_delivery` / `days_remaining_in_window`, `explanation` |
| 4 | `process_refund` | לבצע את הזיכוי — או לסרב ולדרוש הסלמה | `order_id`, `amount`, `reason` | `status`: `APPROVED` / `REJECTED` / `ESCALATION_REQUIRED`, ולצידו `approved_amount`, `refund_id`, `requested_amount`, `auto_refund_cap_usd`, `applicable_policies`, `reasons`, `message` |
| 5 | `submit_resolution` | **שלנו** — רישום הפתרון הסופי וסיום המקרה | `reasoning_chain`, `action_taken`, `customer_response` | אין תוצאה עסקית; זהו אות ה"סיימתי" של הלופ. ראו סעיף 5 |

**שלוש עובדות על החוזה הזה שקובעות את כל שאר העיצוב:**

| העובדה | ההשלכה על הסוכן |
|---|---|
| `process_refund` הוא **הכלי היחיד עם תופעת לוואי** | הוא גם היחיד שמצליבים מולו את ההחלטה הסופית ב-`enforce_resolution` |
| התקרה נאכפת **בתוך הכלי**, לא בפרומפט | בקשה מעל התקרה מחזירה `ESCALATION_REQUIRED`. אי אפשר לשכנע אותו ל-`APPROVED` — ראו סעיף 7 |
| כשל עסקי הוא **דאטה, לא חריגה** | `{"error": "ORDER_NOT_FOUND" / "USER_NOT_FOUND" / "INVALID_AMOUNT", "message": ...}` זורם למודל כ-`tool_result` רגיל |

> 🔧 **לא נגענו בתיאורי הכלים — וזה העיקר.** מדריך הרקע אומר שאם הסוכן בוחר כלי לא
> נכון, התיקון הוא בתיאור הכלי ולא ב-system prompt. התיאורים במארז כבר כתובים לפי
> הכלל הזה: כל אחד אומר גם *מה* הכלי עושה וגם *מתי* לקרוא לו ("Call this first for
> any ticket that mentions an order", "Call this before promising the customer
> anything", "Call it only after check_return_policy reported the claim eligible").
> בדיוק בגלל זה `prompts.py` קצר — הוא מכיל רק מה שתיאור כלי לא יכול לבטא: הסמכות
> של הסוכן וכללי ההתנהגות ששומרים עליו כן לגבי מה שהכלים באמת אמרו.

### דוגמה אמיתית — הכלים בפעולה

טבלה היא תיאור; הנה מה ש**באמת** קורה. כל הערכים למטה נלקחו מקריאה אמיתית לכלים
של המארז, לא נכתבו מהזיכרון. הכלים דטרמיניסטיים ומחשבים תאריכים מול
`reference_date() == 2026-08-05` קבוע (ראו סעיף 5 במטלה), ואין להם תופעות לוואי —
אפשר לשחזר את הכל בלי מפתח API בכלל.

> ⚠️ **מה קבוע ומה לא:** תוצאות הכלים למטה מדויקות ולא משתנות. ה**רצף** הוא ההרצה
> האופיינית שהמודל מייצר — לא pipeline מובטח. כפי שנאמר בסעיף 1, סדר הקריאות הוא
> החלטה של המודל בזמן ריצה.

**🟢 תרחיש 1 — המסלול הנקי (`ORD-1001`)**

פנייה: *"My earbuds from order ORD-1001 arrived cracked right out of the box."*

| # | הקריאה | מה חזר |
|---|---|---|
| 1 | `get_order_details({order_id: "ORD-1001"})` | `user_id=USR-101`, `status=delivered`, `total_amount=35.0`, מצב הפריט `damaged_on_arrival` |
| 2 | `get_user_profile({user_id: "USR-101"})` | `tier=VIP`, `prior_fraud_flags=0`, `lifetime_value=4820.5` |
| 3 | `check_return_policy({order_id, reason: "damaged_on_arrival"})` | `eligible=true`, `verdict=ELIGIBLE`, חלון `45` יום (`11` חלפו, `34` נותרו), `auto_refund_cap_usd=75.0`, `max_refundable_amount=35.0`, `applicable_policies=[POL-RET-02, POL-REF-02]` |
| 4 | `process_refund({order_id, amount: 35.0, reason})` | `status=APPROVED`, `approved_amount=35.0`, `refund_id=RF-1001-3500` |

ואז `submit_resolution`, ומה שחוזר מ-`resolve()`:

```json
{
  "reasoning_chain": [
    "ORD-1001 נמסרה ב-2026-07-25, ופריט אחד (SKU-HDPH-01, 35.00 USD) מסומן damaged_on_arrival.",
    "USR-101 היא לקוחת VIP ללא דגלי הונאה קודמים — חלון 45 יום ותקרה של 75.00 USD.",
    "check_return_policy החזיר ELIGIBLE: 11 יום מהמסירה, 34 נותרו (POL-RET-02, POL-REF-02).",
    "process_refund על 35.00 USD החזיר APPROVED עם refund_id RF-1001-3500."
  ],
  "action_taken": {
    "tools_called": ["get_order_details", "get_user_profile", "check_return_policy", "process_refund"],
    "decision": "AUTO_REFUND_APPROVED",
    "refund_amount": 35.0,
    "refund_id": "RF-1001-3500"
  },
  "customer_response": "היי מאיה, מצטערים שהאוזניות הגיעו סדוקות...",
  "_case_id": "…", "_tool_calls": [ … ], "_validation_warnings": [], "_corrections": [], "_stopped_reason": "stop"
}
```

> 🔧 **זה המבחן של `reasoning_chain`:** כל שורה בו ניתנת להצלבה מול שורה בטבלה
> שמעליה — תאריך, סכום, מזהה סעיף, `refund_id`. זה ההבדל בין שרשרת שאפשר לבקר
> לבין ניסוח כללי שמתאים לכל פנייה.

**🟡 תרחיש 2 — חריגת סמכות (`ORD-1002`)**

שלוש הקריאות הראשונות זהות. ההבדל מתחיל בתקרה:

| הקריאה | מה חזר |
|---|---|
| `get_order_details` | `total_amount=150.0` |
| `get_user_profile` | `tier=Standard` ⬅️ תקרה של `50.0`, לא `75.0` |
| `check_return_policy` | `eligible=true`, `verdict=ELIGIBLE`, `requires_escalation=false`, אבל `auto_refund_cap_usd=50.0` ו-`max_refundable_amount=50.0` |
| `process_refund({amount: 150.0})` | `status=ESCALATION_REQUIRED`, `approved_amount=0.0`, **אין `refund_id`** |

ההודעה מהכלי: *"Refund not issued. 150.00 USD exceeds your automatic authority of
50.00 USD — escalate to a human operations lead."*

> 💡 **שימו לב להבחנה:** `check_return_policy` אמר `ELIGIBLE` ו-
> `requires_escalation=false`. **זכאות וסמכות הן שתי שאלות שונות** — התביעה
> לגיטימית לחלוטין, פשוט מעל מה שהסוכן מורשה לאשר לבד. סוכן שקורא רק את
> `eligible=true` ומדלג על תוצאת `process_refund` יבטיח ללקוח זיכוי שלא קרה.

**🔴 ולמה זה דורש guardrail בקוד שלנו**

מה קורה אם הסוכן מבקש בדיוק את התקרה במקום את הסכום האמיתי? נבדק מול הכלי האמיתי:

```
process_refund({order_id: "ORD-1002", amount: 50.0})
  → status: APPROVED, approved_amount: 50.0, refund_id: "RF-1002-5000"
```

**`APPROVED`.** הכלי לא יכול להבחין בין תביעה כנה על 50$ לבין סוכן שגוזם את הבקשה
כדי לחמוק מהסלמה — התקרה שלו נאכפת, אבל הכוונה לא. בדיוק בשביל זה קיים
`output_tool.py:251-281`: הוא מזהה `requested_amount == cap` שנמוך מסך ההזמנה,
ודורס חזרה ל-`ESCALATION_REQUIRED`. זו הדוגמה הכי חדה לכך ש**guardrail של המארז
לא פוטר אותנו מ-guardrail משלנו**.

**⚫ תרחיש 3 — מלכודת ההזיה (`ORD-2222`)**

קריאה אחת, וזהו:

```
get_order_details({order_id: "ORD-2222"})
  → {"error": "ORDER_NOT_FOUND", "message": "No order found with id 'ORD-2222'."}
```

אין קריאות נוספות. אין תאריך משלוח מומצא, אין `process_refund` על הזמנה שלא
קיימת. `prompts.py` כלל 2 מורה להתייחס למפתח `error` כאות עצירה, וההחלטה יוצאת
`CANNOT_RESOLVE` — בדיוק הסיבה שהערך הרביעי הזה קיים.

## 5. איך כופים פלט מובנה — `submit_resolution` ככלי

לבקש מהמודל להקליד JSON חופשי בסוף ולפרסר אותו ב-regex זו מלכודת שמדריך הרקע
מזהיר ממנה במפורש. במקום זה, צורת הפלט הנדרשת (`reasoning_chain`,
`action_taken`, `customer_response`) מוגדרת ככלי חמישי — `submit_resolution`
(`output_tool.py:59-125`). המודל קורא לו כתור `tool_use` רגיל, כך שהארגומנטים
מגיעים **כבר מאומתים מול סכמה** על ידי ה-API לפני שהקוד שלנו רואה אותם. אין
regex, אין "נקווה שזה מתפרסר".

המודל לא נכפה לקרוא לו מהתור הראשון — הוא עדיין צריך להחליט בעצמו באילו כלים
אמיתיים לחקור. ה-system prompt (`prompts.py:54-59`) אומר לו לקרוא ל-
`submit_resolution` אחרון. **רשת הביטחון** ב-`tool_loop.py:228-240`: אם הלופ
עומד להגיע ל-`max_iterations` בלי שהמודל קרא לו, נעשה תור אחרון עם `tool_choice`
נעול על `submit_resolution` — כך שהסוכן תמיד מסיים עם פלט מובנה תקין במקום להיקטע
באמצע מחשבה.

**ארבעה ערכי החלטה, לא שלושה** (`output_tool.py:36-41`):

| הערך | מתי |
|---|---|
| `AUTO_REFUND_APPROVED` | `process_refund` החזיר `APPROVED` |
| `REJECTED` | התביעה לא זכאית — המדיניות אומרת לא |
| `ESCALATION_REQUIRED` | `process_refund` החזיר `ESCALATION_REQUIRED`, או שהמקרה צריך אדם מסיבה אחרת |
| `CANNOT_RESOLVE` | ההזמנה/המשתמש שנדרשו להחלטה לא נמצאו |

הרביעי קיים במיוחד עבור מלכודת ההזיה: הזמנה שפשוט לא קיימת אינה אישור, אינה
דחייה מדיניותית ואינה הסלמה בגלל תקרה — ודחיסה שלה ל-`REJECTED` הייתה מטשטשת
הבחנה אמיתית בשובל הביקורת.

## 6. שרשרת החשיבה (Reasoning Chain)

`reasoning_chain` הוא רשימת מחרוזות שהמודל מונחה (ב-`prompts.py`, כלל 6) למלא
ב**עובדות קונקרטיות שהוא באמת ראה בתוצאות הכלים** — מזהי הזמנות, סכומים, תאריכים,
מזהי סעיפי מדיניות — ולא ניסוח כללי שיכול להתאים לכל פנייה. זה בדיוק המדד שמדריך
הרקע מציב בסעיף 8: האם השרשרת מסבירה את ההחלטה במספרים ובסעיפים אמיתיים, או שהיא
ניסוח שמתאים לכל מקרה. אפשר לבדוק כל שורה מול `_tool_calls` שחוזר לצידה.

## 7. ⚠️ הפער בין ההחלטה לתשובה — שלוש שכבות הגנה

זו מלכודת ה"פער בין ההחלטה לתשובה" ממדריך הרקע, וההכשלה החמורה ביותר שהמטלה
מציינת: הכלי החזיר `ESCALATION_REQUIRED`, והלקוח קיבל "הזיכוי בוצע". שלוש שכבות
בלתי תלויות מגינות עליה, בכוונה בשלוש רמות שונות:

| # | השכבה | איפה | מה היא עושה |
|---|---|---|---|
| 1 | **פרומפט** | `prompts.py:49-52` | לגזור `decision` ו-`customer_response` מתוצאת הכלי האחרונה בפועל, לא מהכוונה |
| 2 | **סכמה** | `output_tool.validate_schema` (שורה 128) | שדות חסרים או `decision` מחוץ לארבעת הערכים. קריאה לא תקינה מבנית מטופלת בדיוק כמו אי-קריאה בכלל — נזרקת לטובת אותה נסיגה בטוחה (`agent.py:207-213`) |
| 3 | **אכיפה** | `output_tool.enforce_resolution` (שורה 357) | מצליבה את ההחלטה המוצהרת מול מה ש-`process_refund` באמת החזיר — ו**מתקנת** דטרמיניסטית, לא רק מסמנת |

השכבה השלישית היא העיקר. אין קריאת LLM שנייה ואין חישוב מדיניות משוכפל: כל דריסה
היא או קריאה ישירה של שדה `status` / `approved_amount` / `refund_id` של הכלי, או
נסיגה שמרנית ל-`ESCALATION_REQUIRED` / `CANNOT_RESOLVE` כשהמצב עמום. אי-ההתאמה
המקורית נשארת גלויה ב-`_validation_warnings` (מה היה לא בסדר) וב-`_corrections`
(מה שונה ולמה) — שום דבר לא מוסתר בשקט, פשוט כבר לא ייתכן שההודעה השגויה תהיה זו
שהקורא מקבל בפועל.

> 🔧 **guardrail אחד ששווה להצביע עליו:** `output_tool.py:251-281` תופס את המקרה
> שבו הסוכן **משחק את התקרה** — קורא ל-`process_refund` על סכום ששווה בדיוק
> לתקרה, מתחת לסכום ההזמנה האמיתי, כדי לסחוט `APPROVED` במקום
> `ESCALATION_REQUIRED` כן. המקרה הזה נתפס בבדיקה חיה, והתיקון נדרס חזרה להסלמה.
> זה אותו עיקרון שבו `process_refund` עצמו משתמש מול התקרה שלו — **guardrail חי
> בקוד, לא בפרומפט** — רק שכבה אחת החוצה.

## 8. 🔐 זהות הפונה והרשאות חוצות-לקוחות

כברירת מחדל `resolve(ticket_text)` ישלוף כל `order_id`/`user_id` שהמודל מוצא
בטקסט הפנייה, בלי לבדוק אם הוא שייך למי שבאמת הגיש אותה — מתאים להקשר של קונסולת
תפעול פנימית, אבל פער אמיתי בפריסה מול לקוחות.

`resolve(ticket_text, requester_user_id="USR-101")` סוגר אותו. כל תוצאה מוצלחת של
כלי GlobalCart נושאת שדה `user_id` שמזהה את הלקוח הבעלים (נבדק ישירות מול
`mock_services.py` — כל ארבעת הכלים כוללים אותו, לא רק שתי השליפות המובנות מאליהן),
ולכן `agent._authorize_tool_registry()` (`agent.py:60-103`) עוטף את כולם ומחליף
את התוצאה ב-`{"error": "NOT_AUTHORIZED", ...}` כשהבעלים לא תואם — **לפני**
שהמידע האמיתי מגיע להקשר של המודל, לא רק סינון מהטקסט הסופי ללקוח.

> 💡 מכיוון ש-`NOT_AUTHORIZED` משתמש באותה צורת `error` כמו כל כשל עסקי אחר
> בקודבייס, הוא לא דורש טיפול מיוחד בשום מקום אחר: הכלל הקיים ב-`prompts.py`
> ("אם תוצאת כלי מכילה מפתח error") והכלל הקיים ב-`output_tool.py` ("כלי נכשל,
> לא בוצע זיכוי") מכסים אותו בחינם. השמטת `requester_user_id` (ברירת המחדל)
> משחזרת את ההתנהגות הבלתי מוגבלת של היום בדיוק, כך ששום קורא קיים לא נדרש להשתנות.

```bash
python3 run_ticket.py "My order ORD-1001 arrived damaged." USR-999   # לא הבעלים — נדחה
python3 run_ticket.py "My order ORD-1001 arrived damaged." USR-101   # הבעלים — ממשיך רגיל
```

## 9. מקרי קצה ו-Guardrails

| המקרה | איך זה מטופל |
|---|---|
| בקשת זיכוי מעל תקרת האישור האוטומטי | `process_refund` עצמו מסרב ומחזיר `ESCALATION_REQUIRED` — שום ניסוח פרומפט לא יעביר אותו את התקרה. תפקיד הסוכן הוא רק לזהות ולדווח בכנות |
| החזרה מחוץ לחלון הזמן | `check_return_policy` מחזיר `OUTSIDE_RETURN_WINDOW` ומצטט `POL-RET-01`/`POL-RET-02`; הסוכן דוחה ומצטט את מזהה הסעיף |
| הזמנה או משתמש שלא קיימים (מלכודת ההזיה) | הכלים מחזירים `{"error": "ORDER_NOT_FOUND"/"USER_NOT_FOUND", ...}`. ה-system prompt מנחה להתייחס למפתח הזה כאות עצירה ולא לטייח בנתונים מומצאים — `decision` הופך ל-`CANNOT_RESOLVE` |
| קלט פגום (סכום שלילי, סיבה לא חוקית, הזמנה לא קיימת ל-`process_refund`) | הכלים מחזירים `{"error": ...}` מובנה במקום לזרוק חריגה; הסוכן קורא את השגיאה ומדווח במקום לקרוס או לנסות שוב בעיוורון |
| חזרה על אותה קריאת כלי | `tool_loop.py:173-188` עוקב אחרי חתימות `(tool_name, args)` שכבר נראו ומסרב להריץ קריאה זהה, ומחזיר הודעה שאומרת למודל להפסיק לנסות ולפעול לפי מה שכבר יש לו |
| לופ שיוצא משליטה | `max_iterations` (ברירת מחדל 8) חוסם את מספר סבבי הקריאות; אם נגמר, הלופ כופה קריאת `submit_resolution` אחרונה כך שהסוכן עדיין מחזיר תשובה בטוחה ומובנית (ברירת מחדל: הסלמה) במקום להיתקע |
| כשל API/רשת | ה-SDK של Anthropic כבר מנסה שוב שגיאות חיבור ו-408/409/429/5xx פנימית; `tool_loop.py` תופס את מה שמגיע אחרי זה (ניסיונות מוצו, או שגיאה שאינה ניתנת לניסיון חוזר כמו ה-400 של "credit balance too low" שנתפס בבדיקה חיה) ועוטף כ-`ModelAPIError` תוך שמירת עקבות הכלים החלקיות. `ResolverAgent.resolve()` תופס **רק** את השגיאה המוטפסת הזו ומחזיר `ESCALATION_REQUIRED` בטוח במקום להפיל את הקורא — כל חריגה אחרת (באג אמיתי) ממשיכה להתפשט |

> ⚠️ שימו לב לשתי סוגי החריגות ב-`tool_loop.py:39-64`, כי הן אומרות דברים שונים:
> `ToolExecutionError` = באג תכנותי אמיתי (טיפוס ארגומנט שגוי שהגיע לכלי) — **לא**
> נבלע, מתפשט הלאה. `ModelAPIError` = כשל תשתית. כשל עסקי אף פעם אינו חריגה —
> `mock_services.py` מחזיר אותו כ-dict רגיל שזורם למודל כ-tool_result רגיל.
> בדיוק המלכודת "לזרוק exception על בעיה עסקית" ממדריך הרקע.

## 10. 🚀 התקנה והרצה

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# ואז ערכו את .env והגדירו ANTHROPIC_API_KEY
```

**1. בדיקת שפיות למארז (שלא נגענו בו)** — בודק רק את מנוע הדאטה והחוקים, לא את
הסוכן:

```bash
python3 starter-kit/examples/verify_scenarios.py
# All 33 checks passed.
```

**2. פתרון פנייה בודדת:**

```bash
python3 run_ticket.py "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box. Can you sort this out?"
```

מדפיס את תוצאת ה-JSON המובנית המלאה ל-stdout (אזהרות ולידציה, אם יש, הולכות
ל-stderr).

**3. הרצת מערך הרגרסיה המלא** — הסוכן מול כל 9 התרחישים מ-
[`starter-kit/examples/scenarios.md`](starter-kit/examples/scenarios.md)
(תרחישים 5 ו-7 הם שתי הזמנות כל אחד במטלה, ולכן זה מריץ 10 פניות שמכסות את כל
התשעה):

```bash
python3 run_scenarios.py
```

## 11. 🧪 שלוש שכבות בדיקה

יש כאן שלוש רמות אימות, וכל אחת בודקת דבר אחר. **שווה לדעת איזו מהן צריך שינוי
מסוים:**

| המערך | מה נבדק | צריך מפתח API? |
|---|---|---|
| `starter-kit/examples/verify_scenarios.py` | מנוע הדאטה והחוקים (`mock_services.py`, לא נגוע) עקבי פנימית | ❌ |
| `pytest` (`tests/`) | הלוגיקה של `resolver_agent` עצמו — מכניקת הלופ, guardrails, ולידציית פלט | ❌ |
| `run_scenarios.py` | ה**שיפוט** של הסוכן מקצה לקצה מול מודל חי | ✅ |

```bash
pip install -r requirements-dev.txt
pytest
```

מערך ה-`pytest` מכסה את החלקים שלא דורשים קריאת LLM כדי לאמת:
`tests/test_tool_loop.py` מריץ את `run_tool_loop` עם "מודל" מזויף מתוסרט (ראו
`tests/helpers.py`) כדי לאשר את מכניקת הלופ עצמה — מנגנון הקריאה החוזרת, שני
נתיבי הסיום של `max_iterations`, טיפול בכלי לא מוכר ועטיפת `TypeError` — בעוד שכל
קריאת כלי שהמודל המזויף מבקש עדיין מנותבת לפונקציות האמיתיות של `starter-kit`,
לא ל-mock שלהן. `tests/test_output_tool.py` עושה את אותו הדבר ל-
`validate_resolution`, כולל טסט רגרסיה לבאג המדויק של "בקשת חסר כדי לחמוק
מהסלמה" שנתפס בבדיקה חיה.

> 🔧 **איזו שכבה לשינוי שלכם:** שינוי ב-`tool_loop.py` או ב-`output_tool.py`
> ⬅️ טסט `pytest` עם המודל המתוסרט (זול, דטרמיניסטי). שינוי ב-`prompts.py` או
> בתיאור כלי ⬅️ `run_scenarios.py`, כי מה שהשתנה הוא השיפוט של המודל, לא המכניקה.

## 12. 📋 לוגים

`resolver_agent` פולט לוגי JSON מובנים (אובייקט אחד לשורה) ל-**stderr** דרך מודול
`logging` הסטנדרטי — לעולם לא ל-stdout, ששמור לתוצאה האמיתית של `run_ticket.py`.
כל שורת לוג מקריאת `resolve()` אחת נושאת את אותו `case_id`, כך ששורות ממקרים
מקבילים או עוקבים לא מתערבבות. הרמה נשלטת ב-`LOG_LEVEL` (ברירת מחדל `WARNING`,
כך שהרצה מוצלחת נקייה לא מדפיסה כלום ל-stderr):

```bash
LOG_LEVEL=INFO python3 run_ticket.py "..."   2> >(jq .)   # הדפסה יפה של הלוגים בנפרד
```

| רמה | האירוע | מתי |
|---|---|---|
| `DEBUG` | `tool_loop.tool_executed` | כל קריאת כלי GlobalCart אמיתית |
| `WARNING` | `tool_loop.repeat_call_refused` / `unknown_tool_requested` / `max_iterations_reached` | ה-guardrails של הלופ נורים |
| `WARNING` | `agent.resolution_corrected` | ההחלטה נדרסה כדי להתאים למה שהכלי החזיר |
| `WARNING` | `agent.fallback_resolution_used` | המודל לא קרא ל-`submit_resolution`, או שהקריאה לא הייתה תקינה מבנית |
| `ERROR` | `agent.api_error` | קריאת ה-API של Anthropic עצמה נכשלה |
| `INFO` | `agent.case_resolved` | מקרה נפתר נקי, בלי אזהרות |

**לא נרשם בלוג לעולם:** טקסט הפנייה הגולמי או גוף ה-`customer_response` — שניהם
עלולים להכיל את שם הלקוח. שדות הלוג נשארים מבניים: `case_id`, `decision`, מוני
קריאות כלים, `stopped_reason`, שמות טיפוסי שגיאה.

`resolver_agent` לעולם לא קורא ל-`logging.basicConfig()` בעצמו — רק
`run_ticket.py`/`run_scenarios.py` קוראים ל-`configure_logging()`, פעם אחת,
בעלייה. הטמעת החבילה במקום אחר (למשל בחלק ב') אומרת לקרוא לזה בעצמכם, או לא, בלי
שהיא תילחם על ה-root logger.

---

חלק ב' לוקח את אותו סוכן ומפצל אותו לצוות — ו-`tool_loop.py` עובר לשם ללא שינוי.
