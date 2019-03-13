@mainpage MorphStore - Fluffy Morphicorn


<div style="width: 620px; height: 60px; padding-left: 35px;" class="fragment">
    <div style="width: 520px; font-size: 16px; text-indent: -30px">
        "MorphStore. THE Morphing Main Memory Engine" 
    </div>
    <div style="float: right; text-algin: right">
        \- The MorphStore Team, February 2019
    </div>
</div>


This documentation is about __How__ MorphStore works and __How__ to use it

--- 

@section mainpage_roadmap Roadmap to begin with MorphStore
This section is designed to hand a plan to staff members and students, who never worked with MorphStore before, but also as quick reference.

<div class="ToDo">Sinnvolle Struktur für die Doku?</div>
Each section/tutorial is roughly divided into 3 parts:
- **Usage:** How can the described feature be used by the end-user (i.e. the one writing a query)
- **Developer Interface** How can a developer add more functionality to the described feature
- **Conceptual remarks:** Deep dive into the internal workflow/implementation/functionality of this feature. Might only be interesting for other Morph-Store core developers.

According to these 3 parts, code examples are colored differently:
<div class="userCode">
~~~{.cpp}
//This is Code a user can write to do somehting useful, e.g. make a query, generate some data,...
~~~
</div>
<div class="morphStoreDeveloperCode">
~~~{.cpp}
//This is code a developer can use to add functionality to the MorphStore, i.e. add stuff to the code base.
~~~
</div>
<div class="morphStoreBaseCode">
~~~{.cpp}
//This already belongs to the code base of the MorphStore.
~~~
</div>

Tutorials can be found in the appropriate section: \ref tutorials
