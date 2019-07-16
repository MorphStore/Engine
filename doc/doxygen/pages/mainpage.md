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
<br />
<b>Dear SIGMOD 2019 visitors, we are working on the documentation. Please have a little patience.</b>

<div class="box-grid">
<div class="BoxPink" style="grid-column: 1">
<div class=symbol><br /><br />❶</div>
<b>Getting started</b>
<ul>
<li>\ref quickStart</li>
<li>\ref testCases</li>
<li>\ref overview</li>
</ul>
</div>
<div class="BoxBlue" style="grid-column: 2">
<div class=symbol><br /><br />➚</div>
<b>Vector Lib</b>
<ul>
<li>Using the \ref veclib - A short walk through</li>
<li>\ref VectorPrimitives</li>
<li>\ref primitiveTable - A table</li>
<li>VectorLib w/o MorphStore</li>
</ul>
</div>
<div class="BoxBlue" style="grid-column: 1">
<div class=symbol><br />B</div>
<b>Benchmarking</b>
<ul>
<li>\ref variantExecutor</li>
<li>\ref Monitoring</li>
</ul>
</div>
<div class="BoxPink" style="grid-column: 2">
<div class=symbol><br />✎</div>
<b>Writing Queries</b>
<ul>
<li>Hello World!</li>
<li>Available Operators and Compressions</li>
</ul>
</div>

</div>

<div class=howtoread>
<b>How to read our Tutorials</b><br /><br />
Sections/tutorials describe one or more of the following aspects:
- **Usage:** How can the described feature be used by the end-user (i.e. the one writing a query)
- **Developer Interface** How can a developer add more functionality to the described feature
- **Conceptual remarks:** Deep dive into the internal workflow/implementation/functionality of this feature. Might only be interesting for other Morph-Store core developers.
<br /><br />
According to these 3 aspects, code examples are colored differently:
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
<div>
